"""Single-arm legacy equivalence, explicit masks and bounded matched patching."""
import os
os.environ["EXPLORER_NO_MODEL"] = "1"
os.environ["EXPLORER_BUNDLED_CACHE"] = "1"
import sys
import copy
import threading
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))
from runtime import Runtime
from interventions import Plan
from test_adapters import model, tokenizer


def runtime(tmp_path, family="Llama"):
    torch.manual_seed(11)
    rt = Runtime(str(tmp_path), str(tmp_path))
    rt.attach(model(family), tokenizer())
    rt.resid_p50[0] = 1.
    return rt


@pytest.mark.parametrize("family", ["Llama", "Qwen3_5"])
@pytest.mark.parametrize("mode,dose", [("add", .1), ("mul", .7)])
def test_one_arm_matches_legacy_and_restores(family, mode, dose, tmp_path):
    rt = runtime(tmp_path, family)
    messages = [{"role": "user", "content": "t3 t5 t8"}]
    legacy = ["", "", ""]
    for row, delta in rt.steer_multi([(0, 3, dose, mode)], [messages]*3, max_new=3):
        if row is not None:
            legacy[row] += delta
    for row, arm_dose in enumerate([dose, 0. if mode == "add" else 1., -dose]):
        result = rt.generate_one(messages, [(0, 3, arm_dose, mode)], max_new=3)
        assert result["text"] == legacy[row]
        assert result["counts"]["output_tokens"] == len(result["output_token_ids"])
        assert result["finish_reason"] in {"eos", "length"}
        assert rt._push is rt._research_hook is None
    baseline = rt.generate_one(messages, max_new=3)
    assert baseline["text"] == legacy[1]


def test_selected_capture_and_self_patch_are_identity(tmp_path):
    rt = runtime(tmp_path)
    reference = rt.capture_selected("t3 t5 t8", [[0, 0, 3], [1, 2, 4]])
    full = rt.full_activations(reference["prompt_token_ids"])
    assert reference["values"] == [float(full[l,p,n]) for l,p,n in reference["selection"]]
    assert len(reference["values"]) == 2
    base = rt.score_sequence("t3 t5 t8", "t10 t11")
    patched = rt.score_sequence("t3 t5 t8", "t10 t11", patches={
        "reference": reference, "target_token_ids": reference["prompt_token_ids"]})
    assert patched["per_token_logprobs"] == base["per_token_logprobs"]
    bad = copy.deepcopy(reference)
    bad["values"][0] += 1
    with pytest.raises(ValueError, match="hash"):
        rt.score_sequence("t3 t5 t8", "t10", patches={"reference": bad, "target_token_ids": base["prompt_token_ids"]})
    with pytest.raises(ValueError, match="matched"):
        rt.score_sequence("t3 t5", "t10", patches={"reference": reference, "target_token_ids": [3,5]})
    assert rt._research_hook is None


def test_prompt_and_decode_masks_use_absolute_processed_positions(tmp_path):
    rt = runtime(tmp_path)
    plan = Plan(rt, [3, 5, 8], interventions=[dict(layer=0,n=3,dose=2.,mode="mul",
                                                prompt_positions=[1],decode_steps=[0,2])])
    assert [p[4] for p in plan.edits(0, 6)] == [1,3,5]
    assert [p[4] for p in plan.edits(3, 1)] == [3]
    base = rt.score_sequence("t3 t5 t8", "t10 t11")
    inactive = rt.score_sequence("t3 t5 t8", "t10 t11", interventions=[
        dict(layer=0,n=3,dose=0.,mode="mul",prompt_positions=[],decode_steps=[])])
    assert inactive["per_token_logprobs"] == base["per_token_logprobs"]
    changed = rt.score_sequence("t3 t5 t8", "t10 t11", interventions=[
        dict(layer=0,n=3,dose=0.,mode="mul",prompt_positions=[2],decode_steps=[])])
    assert changed["per_token_logprobs"] != base["per_token_logprobs"]
    restored = rt.score_sequence("t3 t5 t8", "t10 t11")
    assert restored["per_token_logprobs"] == base["per_token_logprobs"]


def test_generation_cancel_and_invalid_scopes_restore(tmp_path):
    rt = runtime(tmp_path)
    event = threading.Event()
    event.set()
    result = rt.generate_one([{"role":"user","content":"t3"}], cancel=event)
    assert result["finish_reason"] == "canceled" and result["output_token_ids"] == []
    assert result["cleanup"]["cache_cleared"] and not result["completed"]
    for item in [dict(layer=0,n=3,dose=float("nan"),mode="add"),
                 dict(layer=0,n=3,dose=1.,mode="add",prompt_positions=[99])]:
        with pytest.raises(ValueError):
            rt.score_sequence("t3", "t5", interventions=[item])
        assert rt._research_hook is None


def test_clean_corrupt_selected_patch_and_post_patch_baseline(tmp_path):
    rt = runtime(tmp_path)
    clean, corrupt = "t3 t5 t8", "t3 t6 t8"
    width = rt.adapter.down[0].width
    reference = rt.capture_selected(clean, [[0, 1, n] for n in range(width)], chat=False)
    target = rt.tok(corrupt).input_ids
    baseline = rt.score_sequence(corrupt, " t10 t11", chat=False)
    patched = rt.score_sequence(corrupt, " t10 t11", chat=False,
                                patches={"reference": reference, "target_token_ids": target})
    assert patched["per_token_logprobs"] != baseline["per_token_logprobs"]
    restored = rt.score_sequence(corrupt, " t10 t11", chat=False)
    assert restored["per_token_logprobs"] == baseline["per_token_logprobs"]
    # Matching lengths alone cannot authorize a patch for an undeclared target.
    with pytest.raises(ValueError, match="matched"):
        rt.score_sequence(corrupt, " t10", chat=False, patches={
            "reference": reference, "target_token_ids": reference["prompt_token_ids"]})


def test_uncalibrated_additive_probe_cannot_create_hidden_calibration(tmp_path, monkeypatch):
    rt = runtime(tmp_path)
    rt.resid_p50.clear()
    def forbidden(*args):
        raise AssertionError("A measured request attempted uncharged calibration")
    monkeypatch.setattr(rt, "layer_resid_p50", forbidden)
    with pytest.raises(ValueError, match="explicit calibration"):
        rt.score_sequence("t3 t5", "t8", pushes=[(0, 3, .1, "add")])
    assert rt._research_hook is None
