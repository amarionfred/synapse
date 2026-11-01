"""Sequence likelihood checked against full causal rows and known confounds."""
import os
os.environ["EXPLORER_NO_MODEL"] = "1"
os.environ["EXPLORER_BUNDLED_CACHE"] = "1"
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))
from answer_scoring import contextual_tokens, exact_grade, score_sequence
from runtime import Runtime
from test_adapters import model, tokenizer


@pytest.mark.parametrize("family", ["Llama", "Qwen3_5"])
def test_every_contextual_token_matches_causal_reference(family, tmp_path):
    torch.manual_seed(17)
    rt = Runtime(str(tmp_path), str(tmp_path))
    rt.attach(model(family), tokenizer())
    result = rt.score_sequence("t3 t5", "t8 t10")
    prompt, answer = result["prompt_token_ids"], result["completion_token_ids"]
    with torch.inference_mode():
        logits = rt.adapter.model(input_ids=torch.tensor([prompt + answer]),
                                  use_cache=False, logits_to_keep=0).logits[0]
    reference = [float(torch.log_softmax(logits[len(prompt)-1+i].float(), -1)[token])
                 for i, token in enumerate(answer)]
    assert answer == [8, 10]
    assert result["per_token_logprobs"] == pytest.approx(reference, abs=2e-6)
    assert result["total_logprob"] == pytest.approx(sum(reference), abs=3e-6)
    assert result["mean_logprob"] == pytest.approx(sum(reference)/2, abs=2e-6)
    assert result["counts"]["forward_passes"] == 2
    assert result["cleanup"] == {"interventions_cleared": True, "cache_cleared": True}
    assert rt._research_hook is None
    eos = rt.score_sequence("t3 t5", "t8 t10", eos_policy="include")
    assert eos["completion_token_ids"] == [8, 10, 2]
    assert len(eos["per_token_logprobs"]) == 3
    # Answers share their first token; complete scoring retains the distinction.
    other = rt.score_sequence("t3 t5", "t8 t11")
    assert result["per_token_logprobs"][0] == other["per_token_logprobs"][0]
    assert result["total_logprob"] != other["total_logprob"]


def test_contextual_boundary_merge_is_explicit():
    def encode(text, **kw):
        return SimpleNamespace(input_ids={"a": [1], "ab": [2], "a b": [1, 3]}[text])
    rt = SimpleNamespace(tok=encode, _research_render=lambda text, chat: (text, False))
    with pytest.raises(ValueError, match="retokenizes"):
        contextual_tokens(rt, "a", "b", False)
    assert contextual_tokens(rt, "a", " b", False) == ([1], [3])


def test_partial_cancel_deadline_failure_and_recovery(tmp_path, monkeypatch):
    rt = Runtime(str(tmp_path), str(tmp_path))
    rt.attach(model("Llama"), tokenizer())
    event = threading.Event()
    original = rt._research_eval
    def stop(*args, **kw):
        result = original(*args, **kw)
        event.set()
        return result
    monkeypatch.setattr(rt, "_research_eval", stop)
    result = rt.score_sequence("t3 t5", "t8 t10", cancel=event)
    assert result["finish_reason"] == "canceled" and not result["completed"]
    assert result["counts"]["forward_passes"] == 1
    assert result["per_token_logprobs"] == [] and rt._research_hook is None
    monkeypatch.setattr(rt, "_research_eval", original)
    expired = rt.score_sequence("t3", "t8", deadline=time.monotonic()-1)
    assert expired["finish_reason"] == "deadline" and expired["counts"]["forward_passes"] == 0
    before = rt.score_sequence("t3 t5", "t8 t10")
    def fail(*args, **kw):
        rt._research_hook = "dirty"
        raise RuntimeError("injected forward failure")
    monkeypatch.setattr(rt, "_research_eval", fail)
    with pytest.raises(RuntimeError, match="injected") as caught:
        rt.score_sequence("t3 t5", "t8 t10")
    assert caught.value.research_costs["forward_passes"] == 1
    assert rt._research_hook is None
    monkeypatch.setattr(rt, "_research_eval", original)
    after = rt.score_sequence("t3 t5", "t8 t10")
    assert after["total_logprob"] == before["total_logprob"]


def test_exact_reference_grading_and_validation(tmp_path):
    assert exact_grade("12", "12")
    assert not exact_grade("12 ", "12")
    assert not exact_grade("The answer is 12", "12")
    rt = Runtime(str(tmp_path), str(tmp_path))
    rt.attach(model("Llama"), tokenizer())
    for completion in ("", 3, None):
        with pytest.raises(ValueError):
            rt.score_sequence("t3", completion)
    with pytest.raises(ValueError, match="eos_policy"):
        rt.score_sequence("t3", "t8", eos_policy="first_token_only")
    with pytest.raises(ValueError, match="bounded context"):
        rt.score_sequence(" ".join(["t3"]*128), " t8", chat=False)


def test_queued_deadline_and_cancellation_do_not_load_or_touch_other_request(tmp_path, monkeypatch):
    rt = Runtime(str(tmp_path), str(tmp_path))
    entered, release = threading.Event(), threading.Event()
    def owner():
        with rt.lock:
            rt._research_hook = "owned-by-another-request"
            entered.set()
            assert release.wait(3)
            rt._research_hook = None
    worker = threading.Thread(target=owner)
    worker.start()
    assert entered.wait(2)
    def forbidden():
        raise AssertionError("An expired waiter attempted model loading")
    monkeypatch.setattr(rt, "ensure_loaded", forbidden)
    try:
        result = rt.score_sequence("input", " answer", deadline=time.monotonic()+.05)
        assert result["finish_reason"] == "deadline" and result["execution_started"] is False
        assert result["counts"]["forward_passes"] == 0
        assert rt._research_hook == "owned-by-another-request"
        cancel = threading.Event()
        cancel.set()
        generated = rt.generate_one([{"role": "user", "content": "input"}], cancel=cancel)
        assert generated["finish_reason"] == "canceled" and not generated["execution_started"]
    finally:
        release.set()
        worker.join(2)
