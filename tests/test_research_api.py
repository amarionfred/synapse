"""Independent public research checks; all runtime responses here are synthetic.

No checkpoint is loaded. In particular, the legacy next-token endpoint must keep
its warning for answers that share token zero; that warning is not a complete
answer score or evidence about Qwen's numerical behavior.
"""

import asyncio
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

os.environ["EXPLORER_NO_MODEL"] = "1"
os.environ["EXPLORER_BUNDLED_CACHE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))

import pytest
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient

import server


SCORE_REQUEST = {"text": "Return the count.", "completion": " 10"}
GENERATE_REQUEST = {"messages": [{"role": "user", "content": "Return a digit."}]}


@pytest.fixture
def research_runtime(monkeypatch):
    """Record exactly what crosses the public API; this is not a fake scorer proof."""
    calls = []
    results = {
        "score": {"prompt_token_ids": [20, 21], "completion_token_ids": [1, 2],
                  "per_token_logprobs": [-0.25, -2.0], "total_logprob": -2.25,
                  "mean_logprob": -1.125, "eos_policy": "exclude",
                  "finish_reason": "completed", "completed": True,
                  "identity": {"checkpoint": "synthetic-target", "runtime": "fixture-only"},
                  "counts": {"prompt_tokens": 2, "output_tokens": 2, "requested_tokens": 2,
                             "scored_tokens": 2, "forward_passes": 2, "evaluated_tokens": 3},
                  "cleanup": {"interventions_cleared": True, "cache_cleared": True}},
        "generate": {"text": " 2", "output_token_ids": [4], "prompt_token_ids": [20, 21],
                     "finish_reason": "eos", "stop_token_id": 0, "completed": True,
                     "counts": {"prompt_tokens": 2, "output_tokens": 1,
                                "forward_passes": 2, "evaluated_tokens": 3},
                     "identity": {"checkpoint": "synthetic-target", "runtime": "fixture-only"},
                     "cleanup": {"interventions_cleared": True, "cache_cleared": True}},
    }

    def record(kind, *args, **kwargs):
        assert isinstance(kwargs["cancel"], threading.Event)
        assert not kwargs["cancel"].is_set()
        assert time.monotonic() < kwargs["deadline"] <= time.monotonic() + 301
        calls.append((kind, args, kwargs))
        return results[kind]

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline public fixture attempted real inference")

    fake = SimpleNamespace(
        score_sequence=lambda *a, **k: record("score", *a, **k),
        generate_one=lambda *a, **k: record("generate", *a, **k),
        capabilities=lambda: {"backend": "synthetic-fixture", "research_execution": {"validated": False}},
        calibration_info=lambda: {"source": "synthetic-calibration"},
        ensure_loaded=forbidden,
    )
    monkeypatch.setenv("EXPLORER_PRELOAD", "0")
    monkeypatch.setattr(server, "rt", fake)
    monkeypatch.setattr(server, "NO_MODEL", False)
    return fake, calls, results


def test_research_overview_reports_actual_availability_without_inference(research_runtime, monkeypatch):
    fake, calls, _ = research_runtime
    fake.generate_one = None
    monkeypatch.setattr(server, "NO_MODEL", True)
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
        response = client.get("/api/research/overview")
    assert response.status_code == 200, response.text
    out = response.json()
    assert out["model_key"] == server.MODEL_KEY
    assert out["identity"] == server.IDENTITY
    assert out["calibration"] == {"source": "synthetic-calibration"}
    assert out["operations"]["score"] is True
    assert out["operations"]["generate"] is False
    assert out["live_model_enabled"] is False
    assert out["researcher"] == {"kind": "inspected-local-model", "external_agent": False}
    assert out["live_validation"] == {"validated": False}
    assert out["limits"] == {"request_bytes": 262144, "max_new": 256, "timeout_seconds": 300}
    assert calls == []


def test_research_score_preserves_exact_completion_and_full_sequence_result(research_runtime):
    _, calls, results = research_runtime
    body = {**SCORE_REQUEST, "chat": False, "prefix": "Answer:", "eos_policy": "include",
            "neurons": [{"layer": 2, "n": 3, "dose": -0.5, "mode": "add"}],
            "timeout_seconds": 5}
    results["score"]["eos_policy"] = "include"
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
        response = client.post("/api/research/score", json=body)
    assert response.status_code == 200, response.text
    assert response.json() == results["score"]
    assert len(calls) == 1
    kind, args, kwargs = calls[0]
    assert (kind, args) == ("score", (body["text"], body["completion"]))
    assert {k: v for k, v in kwargs.items() if k not in {"cancel", "deadline"}} == {
        "chat": False, "prefix": "Answer:", "pushes": [(2, 3, -0.5, "add")], "eos_policy": "include",
    }


def test_research_generate_calls_exactly_one_arm_and_preserves_tokens(research_runtime):
    _, calls, results = research_runtime
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
        response = client.post("/api/research/generate", json={**GENERATE_REQUEST, "max_new": 4})
    assert response.status_code == 200, response.text
    assert response.json() == results["generate"]
    assert len(calls) == 1
    kind, args, kwargs = calls[0]
    assert (kind, args) == ("generate", (GENERATE_REQUEST["messages"],))
    assert {k: v for k, v in kwargs.items() if k not in {"cancel", "deadline"}} == {
        "pushes": [], "max_new": 4,
    }


@pytest.mark.parametrize("path,body", [
    ("/api/research/score", SCORE_REQUEST),
    ("/api/research/generate", GENERATE_REQUEST),
])
def test_research_explicit_model_disable_prevents_worker_calls(research_runtime, monkeypatch, path, body):
    _, calls, _ = research_runtime
    monkeypatch.setattr(server, "NO_MODEL", True)
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
        response = client.post(path, json=body)
    assert response.status_code == 422, response.text
    assert "disabled" in response.json()["error"].lower()
    assert calls == []


@pytest.mark.parametrize("path,method,body", [
    ("/api/research/score", "score_sequence", SCORE_REQUEST),
    ("/api/research/generate", "generate_one", GENERATE_REQUEST),
])
def test_research_unavailable_runtime_is_explicit(research_runtime, path, method, body):
    fake, calls, _ = research_runtime
    setattr(fake, method, None)
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
        response = client.post(path, json=body)
    assert response.status_code == 422, response.text
    assert "unavailable" in response.json()["error"].lower()
    assert calls == []


INVALID_SCORE_FIELDS = [
    {"text": ""}, {"text": 12}, {"text": "x" * 32769},
    {"completion": ""}, {"completion": []}, {"prefix": None},
    {"chat": "false"}, {"chat": 1}, {"eos_policy": "first_token"},
    {"heads": [{"layer": 0, "head": 0}]}, {"unexpected": True},
]
INVALID_GENERATE_FIELDS = [
    {"messages": []}, {"messages": "hello"}, {"messages": [{}]},
    {"messages": [{"role": "tool", "content": "A"}]},
    {"messages": [{"role": "user", "content": ""}]},
    {"messages": [{"role": "user", "content": "A", "checkpoint": "other"}]},
    {"messages": [{"role": "user", "content": "A"}] * 33},
    {"max_new": 0}, {"max_new": -1}, {"max_new": 257}, {"max_new": True}, {"max_new": 1.5},
    {"heads": [{"layer": 0, "head": 0}]}, {"unexpected": True},
]
INVALID_SHARED_FIELDS = [
    {"timeout_seconds": 0}, {"timeout_seconds": 301}, {"timeout_seconds": True},
    {"timeout_seconds": 10**500},
    {"timeout_seconds": "60"}, {"neurons": {}},
    {"neurons": [{"layer": 0, "n": 0}] * 33},
    {"neurons": [{"layer": -1, "n": 0}]},
    {"neurons": [{"layer": True, "n": 0}]},
    {"neurons": [{"layer": 0, "n": -1}]},
    {"neurons": [{"layer": 0, "n": 0, "dose": True}]},
    {"neurons": [{"layer": 0, "n": 0, "dose": 10**500}]},
    {"neurons": [{"layer": 0, "n": 0, "mode": "patch"}]},
    {"neurons": [{"layer": 0, "n": 0, "head": 1}]},
]


@pytest.mark.parametrize("path,body", [
    *[("/api/research/score", {**SCORE_REQUEST, **fields})
      for fields in INVALID_SCORE_FIELDS + INVALID_SHARED_FIELDS],
    *[("/api/research/generate", {**GENERATE_REQUEST, **fields})
      for fields in INVALID_GENERATE_FIELDS + INVALID_SHARED_FIELDS],
])
def test_research_invalid_requests_never_execute(research_runtime, path, body):
    _, calls, _ = research_runtime
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
        response = client.post(path, json=body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]
    assert calls == []


@pytest.mark.parametrize("content", [
    b"{", b"[]", b"null", b"true", b"\xff", b" " * 262145,
    b'{"text":"x","completion":"y","timeout_seconds":NaN}',
    b'{"text":"x","completion":"y","neurons":[{"layer":0,"n":0,"dose":Infinity}]}',
])
def test_research_malformed_or_unbounded_json_never_executes(research_runtime, content):
    _, calls, _ = research_runtime
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
        response = client.post("/api/research/score", content=content,
                               headers={"content-type": "application/json"})
    assert response.status_code == 422, response.text
    assert response.json()["error"]
    assert calls == []


def test_research_worker_failure_releases_slot_for_baseline(research_runtime):
    fake, calls, results = research_runtime
    original = fake.score_sequence

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic worker failure after cleanup")

    fake.score_sequence = fail
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
        failed = client.post("/api/research/score", json=SCORE_REQUEST)
        assert failed.status_code == 422, failed.text
        assert failed.json()["error"] == "synthetic worker failure after cleanup"
        fake.score_sequence = original
        restored = client.post("/api/research/score", json={**SCORE_REQUEST, "timeout_seconds": 1})
    assert restored.status_code == 200, restored.text
    assert restored.json() == results["score"]
    assert len(calls) == 1


def test_research_deadline_signals_worker_and_waits_for_cleanup(research_runtime):
    fake, _, _ = research_runtime
    cleaned = threading.Event()

    def bounded_worker(*args, cancel, deadline, **kwargs):
        try:
            assert cancel.wait(3), "Server did not signal the request deadline"
            assert time.monotonic() >= deadline
            return {"finish_reason": "deadline", "output_token_ids": [4], "partial": True}
        finally:
            cleaned.set()

    fake.generate_one = bounded_worker
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
        response = client.post("/api/research/generate", json={
            **GENERATE_REQUEST, "timeout_seconds": 0.05,
        })
        assert cleaned.is_set(), "Response acknowledged before the worker cleaned up"
    assert response.status_code == 200, response.text
    assert response.json() == {"finish_reason": "deadline", "output_token_ids": [4], "partial": True}


def test_research_cancelled_caller_holds_shared_slot_until_worker_cleanup(research_runtime):
    """Exercise the server's request boundary; no model state is simulated as proof."""
    started, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    second_entered = threading.Event()
    seen_cancel = []

    def first_worker(*args, cancel, **kwargs):
        seen_cancel.append(cancel)
        started.set()
        try:
            assert release.wait(5), "Fixture cleanup was never released"
        finally:
            cleaned.set()

    def second_worker(*args, **kwargs):
        second_entered.set()
        assert cleaned.is_set(), "Another role entered before previous cleanup"
        return "restored baseline"

    async def scenario():
        gate = asyncio.Lock()

        async def is_disconnected():
            return False

        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(research_gate=gate)),
                                  is_disconnected=is_disconnected)
        first = asyncio.create_task(server._research_call(request, first_worker, timeout=10))
        try:
            assert await asyncio.to_thread(started.wait, 3)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert seen_cancel[0].is_set()
            assert gate.locked(), "Cancelled caller released the slot while the worker was active"
            second = asyncio.create_task(server._research_call(request, second_worker, timeout=5))
            await asyncio.sleep(0)
            assert not second_entered.is_set()
            release.set()
            assert await asyncio.wait_for(second, 5) == "restored baseline"
            assert cleaned.is_set() and not gate.locked()
        finally:
            release.set()

    asyncio.run(scenario())


def test_research_queued_deadline_never_starts_worker_or_releases_another_slot(research_runtime):
    calls = []

    async def scenario():
        gate = asyncio.Lock()
        await gate.acquire()

        async def is_disconnected():
            return False

        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(research_gate=gate)),
                                  is_disconnected=is_disconnected)
        try:
            with pytest.raises(HTTPException) as error:
                await server._research_call(request, lambda **kwargs: calls.append("expired"), timeout=0.01)
            assert error.value.status_code == 408
            assert gate.locked() and calls == []
        finally:
            gate.release()
        result = await server._research_call(request, lambda **kwargs: "next baseline", timeout=1)
        assert result == "next baseline" and not gate.locked()

    asyncio.run(scenario())


def test_research_disconnected_waiter_never_enters_worker(research_runtime):
    calls = []

    async def scenario():
        gate = asyncio.Lock()

        async def is_disconnected():
            return True

        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(research_gate=gate)),
                                  is_disconnected=is_disconnected)
        with pytest.raises(HTTPException) as error:
            await server._research_call(request, lambda **kwargs: calls.append("stale"), timeout=1)
        assert error.value.status_code == 499
        assert calls == [] and not gate.locked()

    asyncio.run(scenario())


def test_research_running_disconnect_signals_cancel_before_slot_reuse(research_runtime):
    started, cleaned = threading.Event(), threading.Event()

    def worker(*args, cancel, **kwargs):
        started.set()
        try:
            assert cancel.wait(3), "Disconnected request did not cancel its worker"
            return {"finish_reason": "canceled", "completed": False}
        finally:
            cleaned.set()

    async def scenario():
        gate = asyncio.Lock()

        async def is_disconnected():
            return started.is_set()

        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(research_gate=gate)),
                                  is_disconnected=is_disconnected)
        out = await server._research_call(request, worker, timeout=5)
        assert out == {"finish_reason": "canceled", "completed": False}
        assert cleaned.is_set() and not gate.locked()

    asyncio.run(scenario())


class AnswerTokens:
    """The stand-alone answers collide at token zero, by construction."""

    def __call__(self, text, **kwargs):
        assert kwargs.get("add_special_tokens") is False
        return SimpleNamespace(input_ids={" 10": [1, 2], " 11": [1, 3], " 2": [4]}[text])

    def decode(self, ids):
        return "".join({1: " 1", 2: "0", 3: "1", 4: " 2"}.get(int(i), "?") for i in ids)


@pytest.fixture
def legacy_runtime(monkeypatch):
    calls = []
    base = torch.log_softmax(torch.arange(128, dtype=torch.float32) / 10, dim=0)
    changed = torch.log_softmax(torch.arange(127, -1, -1, dtype=torch.float32) / 10, dim=0)

    def next_logprobs(text, chat, prefix, pushes, heads):
        calls.append((text, chat, prefix, pushes, heads))
        return base, changed if pushes else None

    def forbidden(*args, **kwargs):
        raise AssertionError("Endpoint fixture attempted to load a checkpoint")

    fake = SimpleNamespace(tok=AnswerTokens(), next_logprobs=next_logprobs, ensure_loaded=forbidden)
    monkeypatch.setattr(server, "rt", fake)
    # This only exposes the synthetic runtime; startup remains in NO_MODEL mode.
    monkeypatch.setattr(server, "_live_off", lambda: None)
    return fake, calls, base, changed


def test_legacy_shared_first_token_warning_remains_explicit(legacy_runtime):
    _, calls, base, _ = legacy_runtime
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
        response = client.post("/api/agent/logprobs", json={
            "text": "Compute the exact answer:", "chat": False, "prefix": "Answer:",
            "targets": [" 10", " 11", " 2"], "k": 999,
        })
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result["top"]) == 100
    first, second, single = result["targets"]
    assert first["text"] != second["text"]
    assert first["token"] == second["token"] == " 1"
    assert first["first_token_only"] is second["first_token_only"] is True
    assert first["logprob"] == second["logprob"] == round(float(base[1]), 3)
    assert "first_token_only" not in single
    assert calls == [("Compute the exact answer:", False, "Answer:", [], [])]


def test_legacy_paired_logprobs_preserve_both_arms_and_provenance(legacy_runtime):
    _, calls, base, changed = legacy_runtime
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
        response = client.post("/api/agent/logprobs", json={
            "text": "Compute:", "targets": [" 10", " 11"],
            "neurons": [{"layer": 2, "n": 3, "dose": -0.5, "mode": "add"}],
        })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["baseline"]["targets"][0]["logprob"] == round(float(base[1]), 3)
    assert result["intervened"]["targets"][0]["logprob"] == round(float(changed[1]), 3)
    assert result["kl"] == pytest.approx(float((base.exp() * (base - changed)).sum()), abs=0.0001)
    assert result["pushed"] == [{"neuron": "L2 N3", "dose": -0.5, "mode": "add"}]
    assert result["heads"] == []
    assert calls == [("Compute:", True, "", [(2, 3, -0.5, "add")], [])]


def test_legacy_generation_preserves_three_rows_without_extra_model_work(legacy_runtime):
    fake, calls, _, _ = legacy_runtime
    generated = []

    def steer_multi(pushes, messages, max_new):
        generated.append((pushes, messages, max_new))
        yield 0, "up-"
        yield 1, "base-"
        yield 2, "down-"
        yield 0, "answer"
        yield 1, "answer"
        yield 2, "answer"
        yield None, None

    fake.steer_multi = steer_multi
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
        response = client.post("/api/agent/steer", json={
            "prompt": "Return a digit", "layer": 2, "n": 3, "dose": 0.5, "max_new": 4,
        })
    assert response.status_code == 200, response.text
    result = response.json()
    assert [result[key] for key in ("up", "baseline", "down")] == [
        "up-answer", "base-answer", "down-answer",
    ]
    assert result["steered"] == [{"neuron": "L2 N3", "dose": 0.5, "mode": "add"}]
    assert generated == [([(2, 3, 0.5, "add")], [[{"role": "user", "content": "Return a digit"}]] * 3, 4)]
    assert calls == []


@pytest.mark.parametrize("path,body", [
    ("/api/agent/logprobs", {"text": "Compute", "targets": [" 10", " 11"]}),
    ("/api/agent/steer", {"prompt": "Compute", "layer": 0, "n": 0}),
])
def test_disabled_live_legacy_routes_do_not_reach_runtime(monkeypatch, path, body):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled live request reached runtime")

    monkeypatch.setattr(server, "rt", SimpleNamespace(
        next_logprobs=forbidden, steer_multi=forbidden, ensure_loaded=forbidden,
    ))
    monkeypatch.setattr(server, "NO_MODEL", True)
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
        response = client.post(path, json=body)
    assert response.status_code == 200
    assert "EXPLORER_NO_MODEL=1" in response.json()["error"]


def test_public_persisted_jobs_and_presets_share_compact_state(tmp_path, monkeypatch):
    from test_experiment_runner import FakeRuntime, spec
    rt = FakeRuntime()
    monkeypatch.setattr(server, "rt", rt)
    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(server, "NO_MODEL", False)
    monkeypatch.setattr(server, "_research_runner", None)
    monkeypatch.setenv("EXPLORER_PRELOAD", "0")
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
        saved = client.post("/api/research/presets", json={"neurons": [{"layer": 0, "n": 3, "dose": .25}]} )
        assert saved.status_code == 201, saved.text
        preset = saved.json()
        assert client.post("/api/research/presets/replay", json={"id": preset["id"]}).json() == spec()["treatment"]
        submitted = client.post("/api/research/jobs", json=spec())
        assert submitted.status_code == 202, submitted.text
        job = submitted.json()
        done = server._research_runner.wait(job["id"])
        assert done["state"] == "completed"
        listed = client.get("/api/research/jobs").json()["items"]
        assert listed[0]["id"] == job["id"]
        feedback = client.get(f'/api/research/jobs/{job["id"]}/feedback').json()
        assert feedback["outcomes"]["wins"] == 1
        assert "pairs" not in feedback and "items" not in feedback
        assert client.get("/api/research/jobs/no-such-job").status_code == 404
        final = client.post("/api/research/jobs", json=spec(split="final", candidate_frozen=True)).json()
        server._research_runner.wait(final["id"])
        assert client.get(f'/api/research/jobs/{final["id"]}/feedback').status_code == 422
    assert {p.name for p in tmp_path.iterdir()} <= {
        "research.sqlite3", "research.sqlite3-wal", "research.sqlite3-shm", "research-runner.lock"}


def test_public_researcher_runs_inspected_model_and_persists_frozen_episode(tmp_path, monkeypatch):
    from test_local_researcher import FakeRuntime, MENU, BUDGET, RESEARCHER, TARGET
    rt = FakeRuntime()
    rt.research_identity = lambda: {"fixture": True}
    monkeypatch.setattr(server, "rt", rt)
    monkeypatch.setattr(server, "STATE", tmp_path)
    monkeypatch.setattr(server, "NO_MODEL", False)
    monkeypatch.setattr(server, "_researchers", None)
    monkeypatch.setenv("EXPLORER_PRELOAD", "0")
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
        created = client.post("/api/research/researchers", json={"menu": MENU, "brief": "Frozen",
            "researcher_config": RESEARCHER, "target_config": TARGET, "budget": BUDGET})
        assert created.status_code == 201, created.text
        ident = created.json()["id"]
        result = client.post(f'/api/research/researchers/{ident}/choose', json={"timeout_seconds": 5})
        assert result.status_code == 200, result.text
        assert result.json()["action_id"] == "good"
        assert len(rt.calls) == 2 and all(c["pushes"] == RESEARCHER["pushes"] for c in rt.calls)
        assert client.get(f'/api/research/researchers/{ident}').json()["snapshot"]["cost"]["requests"] == 2
        assert client.post(f'/api/research/researchers/{ident}/choose', json={"brief": "adaptive hint"}).status_code == 422
        invalid = client.post(f'/api/research/researchers/{ident}/choose', json={"feedback": [
            {"split": "validation", "action_id": "good", "correct": 1, "total": 1}]})
        assert not invalid.json()["completed"] and len(rt.calls) == 2
        assert client.get('/api/research/researchers/missing').status_code == 404
