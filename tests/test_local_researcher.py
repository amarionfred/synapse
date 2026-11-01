"""Measured menu-selection fixtures exercise actual runtime request contents."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))
from local_researcher import LocalResearcher, ResearchLimit


MENU = [{"id": "bad", "description": "A fixed control", "pushes": [[1, 1, -.1, "add"]]},
        {"id": "good", "description": "A fixed candidate", "pushes": [[1, 2, .1, "add"]]}]
BUDGET = {"max_requests": 16, "max_tokens": 10000, "max_seconds": 10, "max_decisions": 4}
RESEARCHER = {"pushes": [[0, 1, .2, "add"]]}
TARGET = {"pushes": [[0, 2, .3, "add"]]}


class FakeRuntime:
    def __init__(self):
        self.lock = threading.RLock()
        self.calls = []
        self.failure = None

    def capabilities(self):
        return {"context_tokens": 16}

    def research_token_bound(self, operation, **request):
        return 32

    def score_sequence(self, **kwargs):
        self.calls.append(deepcopy({k: v for k, v in kwargs.items() if k not in ("cancel", "deadline")}))
        if self.failure == "raise":
            raise RuntimeError("fixture forward failed")
        menu_json = kwargs["text"].split("\nAction menu: ", 1)[1].split("\nDevelopment feedback: ", 1)[0]
        mapping = json.loads(menu_json)
        action = next(item["action_id"] for item in mapping if item["display_id"] == kwargs["completion"].strip())
        # Both actions share their first token and first-token logprob. Only
        # complete teacher-forced scoring can distinguish them.
        logprobs = [-.4, -.1 if action == "good" else -4.0]
        completed = self.failure != "cancel"
        if self.failure == "cancel":
            kwargs["cancel"].set()
        return {"completed": completed, "finish_reason": "completed" if completed else "canceled",
                "completion_token_ids": [10, 11 if action == "good" else 12],
                "per_token_logprobs": logprobs, "total_logprob": sum(logprobs),
                "counts": {"evaluated_tokens": 3, "forward_passes": 2},
                "identity": {"checkpoint": "fixture"}}

    def generate_one(self, **kwargs):
        self.calls.append(deepcopy({k: v for k, v in kwargs.items() if k not in ("cancel", "deadline")}))
        return {"text": "9", "output_token_ids": [9], "completed": True, "finish_reason": "length",
                "counts": {"evaluated_tokens": 2, "forward_passes": 1}}


def researcher(runtime=None, budget=None, **kwargs):
    return LocalResearcher(runtime or FakeRuntime(), RESEARCHER, TARGET, budget or BUDGET, **kwargs)


def test_complete_scores_choose_semantics_across_menu_and_display_permutations():
    for seed in range(6):
        rt = FakeRuntime()
        local = researcher(rt)
        result = local.choose(list(reversed(MENU)) if seed % 2 else MENU, seed=seed,
                              display_ids=["11", "12"] if seed % 2 else ["A", "B"])
        assert result["completed"] and result["action_id"] == "good"
        assert len(rt.calls) == 2
        assert all(len(row["completion_token_ids"]) == 2 for row in result["scores"])
        assert result["cost"]["requests"] == result["cost"]["researcher_requests"] == 2
        assert result["cost"]["actual_tokens"] == result["cost"]["charged_tokens"] == 6
        assert result["cost"]["target_requests"] == 0


def test_actual_request_custody_and_roles_are_separate():
    rt = FakeRuntime()
    local = researcher(rt)
    feedback = [{"split": "development", "action_id": "bad", "correct": 1, "total": 4}]
    assert local.choose(MENU, brief="Fixed task brief", feedback=feedback)["completed"]
    for call in rt.calls:
        assert call["pushes"] == RESEARCHER["pushes"]
        assert "Fixed task brief" in call["text"]
        assert "correct" in call["text"] and "reference_answer" not in call["text"]
        assert call["eos_policy"] == "exclude"
    output = local.generate_target("good", [{"role": "user", "content": "Return the final digit."}], max_new=1)
    assert output["text"] == "9"
    assert rt.calls[-1]["pushes"] == TARGET["pushes"] + MENU[1]["pushes"]
    starts = [e for e in local.snapshot()["events"] if e["event"] == "request_started"]
    assert starts[0]["target_config"] == TARGET
    assert starts[0]["researcher_config"] == RESEARCHER
    assert starts[-1]["target_config"]["pushes"] == rt.calls[-1]["pushes"]
    assert local.snapshot()["cost"]["target_requests"] == 1


@pytest.mark.parametrize("feedback", [[{"split": "final", "action_id": "good", "correct": 1, "total": 1}],
                                      [{"split": "validation", "action_id": "good", "correct": 1, "total": 1}],
                                      [{"split": "development", "action_id": "good", "correct": 1, "total": 1,
                                        "reference_answer": "SECRET_FINAL_LABEL"}]])
def test_forbidden_feedback_never_reaches_proposer(feedback):
    rt = FakeRuntime()
    local = researcher(rt)
    result = local.choose(MENU, feedback=feedback)
    assert not result["completed"] and result["action_id"] is None
    assert rt.calls == []
    assert local.snapshot()["decisions"] == 1


def test_frozen_protocol_rejects_adaptive_external_hints():
    rt = FakeRuntime()
    local = researcher(rt)
    assert local.choose(MENU, brief="Frozen")["completed"]
    result = local.choose(MENU, brief="Frozen plus external adaptive hint")
    assert result["finish_reason"] == "failed" and "frozen" in result["error"]
    assert len(rt.calls) == 2
    changed = deepcopy(MENU)
    changed[0]["description"] = "External replacement"
    assert not local.choose(changed, brief="Frozen")["completed"]
    assert len(rt.calls) == 2


def test_failures_remain_charged_and_resume_does_not_reset_budget():
    rt = FakeRuntime()
    rt.failure = "raise"
    budget = {**BUDGET, "max_requests": 2, "max_tokens": 64}
    local = researcher(rt, budget)
    assert not local.choose(MENU)["completed"]
    snapshot = local.snapshot()
    assert snapshot["cost"]["requests"] == 1
    assert snapshot["cost"]["charged_tokens"] == 32
    assert snapshot["cost"]["unknown_cost_requests"] == 1
    # Summaries are not authority: resume recomputes charges from journal facts.
    snapshot["cost"]["requests"] = snapshot["cost"]["charged_tokens"] = 0
    resumed = researcher(rt, budget, state=snapshot)
    assert resumed.snapshot()["cost"]["charged_tokens"] == 32
    assert not resumed.choose(MENU)["completed"]
    assert not resumed.choose(MENU)["completed"]
    assert len(rt.calls) == 2
    assert resumed.snapshot()["cost"]["charged_tokens"] == 64


def test_partial_cancellation_is_charged_and_never_selects_an_action():
    rt, cancel = FakeRuntime(), threading.Event()
    rt.failure = "cancel"
    local = researcher(rt)
    result = local.choose(MENU, cancel=cancel)
    assert result["finish_reason"] == "canceled" and result["action_id"] is None
    assert result["cost"]["requests"] == 1 and result["cost"]["actual_tokens"] == 3
    assert rt.lock.acquire(blocking=False)
    rt.lock.release()
    rt.failure = None
    cancel.clear()
    assert local.choose(MENU, cancel=cancel)["completed"]


def test_prospective_budget_prevents_unaffordable_requests():
    rt = FakeRuntime()
    local = researcher(rt, {**BUDGET, "max_tokens": 31})
    assert local.choose(MENU)["finish_reason"] == "token_budget"
    assert rt.calls == []
    local = researcher(rt, {**BUDGET, "max_requests": 1})
    result = local.choose(MENU)
    assert result["finish_reason"] == "request_budget" and result["action_id"] is None
    assert len(rt.calls) == 1 and len(result["scores"]) == 1


def test_deadline_prevents_forward_and_does_not_consume_unknown_request_cost():
    rt = FakeRuntime()
    local = researcher(rt)
    result = local.choose(MENU, deadline=time.monotonic() - 1)
    assert result["finish_reason"] == "deadline" and rt.calls == []
    assert local.snapshot()["decisions"] == 1


def test_target_rejects_final_access_and_overlapping_configuration():
    rt = FakeRuntime()
    local = researcher(rt)
    assert local.choose(MENU)["completed"]
    with pytest.raises(ValueError, match="development"):
        local.generate_target("good", [{"role": "user", "content": "final input"}], split="final")
    assert len(rt.calls) == 2
    overlapping = [{"id": "same", "description": "conflict", "pushes": TARGET["pushes"]}]
    local = researcher(rt)
    assert local.choose(overlapping)["completed"]
    with pytest.raises(ValueError, match="Overlapping"):
        local.generate_target("same", [{"role": "user", "content": "test"}])


def test_record_callback_sees_reservation_before_inference():
    rt = FakeRuntime()
    seen = []

    def record(event):
        if event["event"] == "request_started":
            assert len(rt.calls) == event["request_id"] - 1
        seen.append(event)

    local = researcher(rt, record=record)
    assert local.choose(MENU)["completed"]
    assert [e["event"] for e in seen] == ["decision_started", "request_started", "request_finished", "request_started", "request_finished", "decision"]


def test_durable_episode_freezes_protocol_and_retains_costs_after_reopen(tmp_path):
    from local_researcher import ResearcherStore
    rt = FakeRuntime()
    rt.research_identity = lambda: {"checkpoint": {"fixture": True}, "runtime": {"version": 1}, "calibration": {}}
    store = ResearcherStore(rt, tmp_path)
    protocol = {"menu": MENU, "brief": "Fixed task", "researcher_config": RESEARCHER,
                "target_config": TARGET, "budget": BUDGET}
    try:
        episode = store.create(protocol)
        assert rt.calls == []
        first = store.run(episode["id"])
        assert first["completed"] and first["action_id"] == "good"
        saved = store.get(episode["id"])
        assert saved["snapshot"]["cost"]["requests"] == 2
        assert saved["protocol"]["brief"] == "Fixed task"
        store.close()
        store = ResearcherStore(rt, tmp_path)
        second = store.run(episode["id"])
        assert second["cost"]["requests"] == 4 and second["decision"] == 2
        assert len(rt.calls) == 4
        rejected = store.run(episode["id"], [{"split": "final", "action_id": "good", "correct": 1, "total": 1}])
        assert not rejected["completed"] and len(rt.calls) == 4
        rt.research_identity = lambda: {"different": True}
        with pytest.raises(ValueError, match="identity changed"):
            store.run(episode["id"])
        assert len(rt.calls) == 4
    finally:
        store.close()


def test_unknown_native_request_is_charged_before_explicit_continuation(tmp_path):
    from local_researcher import ResearcherStore
    rt = FakeRuntime()
    rt.research_identity = lambda: {"fixture": True}
    store = ResearcherStore(rt, tmp_path)
    episode = store.create({"menu": MENU, "researcher_config": RESEARCHER,
                            "target_config": TARGET, "budget": BUDGET})
    # Simulate process death after committing a reservation, before a result.
    snap = episode["snapshot"]
    snap["events"] = [{"index": 0, "event": "decision_started", "decision": 1},
                      {"index": 1, "event": "request_started", "request_id": 1,
                       "role": "researcher", "token_reservation": 32}]
    store._db.execute("UPDATE research_episodes SET snapshot=?,state='running' WHERE id=?",
                      (json.dumps(snap), episode["id"]))
    store._db.commit()
    store.close()
    store = ResearcherStore(rt, tmp_path)
    try:
        assert store.get(episode["id"])["state"] == "interrupted"
        assert rt.calls == []  # Opening the store never replays work.
        result = store.run(episode["id"])
        assert result["decision"] == 2
        assert result["cost"]["requests"] == 3
        assert result["cost"]["unknown_cost_requests"] == 1
        assert result["cost"]["charged_tokens"] == 38
    finally:
        store.close()
