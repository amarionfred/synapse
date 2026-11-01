"""Deterministic offline outcome, custody, interruption and budget checks."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys
import threading
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))
from experiment_schema import experiment_spec, grade_exact
from experiment_runner import ExperimentRunner

IDENTITY = {"checkpoint": {"model_key": "fixture", "identity": {"model": "synthetic"}},
            "runtime": {"backend": "fixture", "implementation": "research-v1"},
            "calibration": {"fixture": True}}


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.identity = deepcopy(IDENTITY)
        self.fail_once = False
        self.active = self.peak = 0

    def research_identity(self):
        return deepcopy(self.identity)

    def capabilities(self):
        return {"context_tokens": 64}

    def generate_one(self, messages, *, pushes, max_new, cancel, deadline):
        self.calls.append({"messages": deepcopy(messages), "pushes": deepcopy(pushes),
                           "max_new": max_new, "deadline": deadline})
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("fixture inference failure")
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            changed = bool(pushes)
            return {"text": " 9\n" if changed else "12", "output_token_ids": [9] if changed else [1, 2],
                    "prompt_token_ids": [3, 4], "finish_reason": "eos", "completed": True,
                    "cleanup": {"interventions_cleared": True, "cache_cleared": True},
                    "counts": {"prompt_tokens": 2, "output_tokens": 1 if changed else 2,
                               "forward_passes": 1, "evaluated_tokens": 2}, "identity": self.research_identity()}
        finally:
            self.active -= 1

    def score_sequence(self, text, completion, *, pushes, chat, prefix, eos_policy, cancel, deadline):
        self.calls.append({"text": text, "completion": completion, "pushes": deepcopy(pushes)})
        return {"prompt_token_ids": [3, 4], "completion_token_ids": [1, 2],
                "per_token_logprobs": [-.2, -.1] if pushes else [-.2, -.8],
                "total_logprob": -.3 if pushes else -1., "mean_logprob": -.15 if pushes else -.5,
                "completed": True, "finish_reason": "completed",
                "cleanup": {"interventions_cleared": True, "cache_cleared": True},
                "counts": {"prompt_tokens": 2, "output_tokens": 0, "scored_tokens": 2,
                           "forward_passes": 2, "evaluated_tokens": 5}}


def spec(**changes):
    value = {"split": "development", "items": [{"id": "a", "prompt": "Run the hidden rule", "answer": "9"}],
             "baseline": {"pushes": []}, "treatment": {"pushes": [[0, 3, .25, "add"]]},
             "max_new": 8, "limits": {"max_requests": 8, "max_tokens": 512, "max_wall_seconds": 30}}
    value.update(changes)
    return value


def test_paired_exact_results_and_real_generation_custody(tmp_path):
    rt = FakeRuntime()
    runner = ExperimentRunner(rt, tmp_path)
    try:
        original = spec()
        job = runner.submit(original)
        original["items"][0]["answer"] = "12"  # Submitted specs are frozen, not caller-owned views.
        done = runner.wait(job["id"])
        assert done["state"] == "completed" and done["cleanup_acknowledged"]
        assert done["outcomes"] == {"paired_items": 1, "wins": 1, "losses": 0, "ties": 0,
                                     "regressions": 0, "baseline_correct": 0, "treatment_correct": 1,
                                     "metric": "exact_answer_accuracy"}
        assert done["costs"]["requests"] == 2 and done["costs"]["charged_tokens"] == 7
        assert done["costs"]["observed_tokens"] == 7 and done["costs"]["unknown_cost_requests"] == 0
        assert done["costs"]["forward_passes"] == 2 and done["costs"]["evaluated_tokens"] == 4
        assert done["configurations"] == {"baseline": {"pushes": []}, "treatment": {"pushes": [[0, 3, .25, "add"]]}}
        assert [c["messages"] for c in rt.calls] == [[{"role": "user", "content": "Run the hidden rule"}]] * 2
        assert all("answer" not in c and "completion" not in c for c in rt.calls)
        feedback = runner.feedback(job["id"])
        assert set(feedback) == {"id", "state", "split", "spec_hash", "identity", "outcomes", "costs"}
        assert "items" not in feedback and "pairs" not in feedback
        assert {p.name for p in tmp_path.iterdir()} <= {"research.sqlite3", "research.sqlite3-wal", "research.sqlite3-shm", "research-runner.lock"}
    finally:
        assert runner.shutdown()


def test_score_counts_all_tokens_and_is_not_accuracy(tmp_path):
    rt = FakeRuntime()
    runner = ExperimentRunner(rt, tmp_path)
    try:
        done = runner.wait(runner.submit(spec(mode="sequence_score"))["id"])
        assert done["state"] == "completed"
        assert done["outcomes"]["accuracy_measured"] is False
        assert done["outcomes"]["mean_delta_total_logprob"] == pytest.approx(.7)
        assert done["costs"]["charged_tokens"] == 8  # scored_tokens must not be hidden by output_tokens=0.
        assert rt.calls[0]["completion"] == "9"
    finally:
        runner.shutdown()


def test_validation_and_final_summaries_never_feed_researcher(tmp_path):
    runner = ExperimentRunner(FakeRuntime(), tmp_path)
    try:
        for split in ("validation", "final"):
            job = runner.submit(spec(split=split, candidate_frozen=True))
            runner.wait(job["id"])
            with pytest.raises(ValueError, match="development"):
                runner.feedback(job["id"])
        with pytest.raises(ValueError, match="frozen"):
            runner.submit(spec(split="final"))
    finally:
        runner.shutdown()


def test_request_and_token_budgets_do_not_reset_on_failure_resume(tmp_path):
    rt = FakeRuntime()
    rt.fail_once = True
    runner = ExperimentRunner(rt, tmp_path)
    try:
        job = runner.submit(spec(limits={"max_requests": 2, "max_tokens": 128, "max_wall_seconds": 30}))
        failed = runner.wait(job["id"])
        assert failed["state"] == "failed" and failed["costs"]["requests"] == 1
        assert failed["costs"]["charged_tokens"] == 64 and failed["costs"]["unknown_cost_requests"] == 1
        runner.resume(job["id"])
        exhausted = runner.wait(job["id"])
        assert exhausted["state"] == "budget_exhausted"
        assert exhausted["costs"]["requests"] == 2 and len(rt.calls) == 2
        assert exhausted["progress"]["completed_arms"] == 1
        assert exhausted["costs"]["charged_tokens"] == 68
        with pytest.raises(ValueError, match="Only"):
            runner.resume(job["id"])
    finally:
        runner.shutdown()


def test_insufficient_token_reservation_prevents_any_request(tmp_path):
    rt = FakeRuntime()
    runner = ExperimentRunner(rt, tmp_path)
    try:
        done = runner.wait(runner.submit(spec(limits={"max_requests": 8, "max_tokens": 63, "max_wall_seconds": 30}))["id"])
        assert done["state"] == "budget_exhausted" and done["costs"]["requests"] == 0
        assert rt.calls == []
    finally:
        runner.shutdown()


def test_cancel_acknowledged_only_after_runtime_cleanup_then_resume(tmp_path):
    class Slow(FakeRuntime):
        def __init__(self):
            super().__init__()
            self.entered, self.release, self.cleaned = threading.Event(), threading.Event(), threading.Event()
            self.first = True

        def generate_one(self, messages, *, cancel, **kwargs):
            if not self.first:
                return super().generate_one(messages, cancel=cancel, **kwargs)
            self.first = False
            self.calls.append({"messages": messages})
            self.entered.set()
            assert self.release.wait(5)
            self.cleaned.set()
            return {"text": "", "completed": False, "finish_reason": "canceled",
                    "cleanup": {"interventions_cleared": True, "cache_cleared": True},
                    "counts": {"prompt_tokens": 2, "output_tokens": 0}}

    rt = Slow()
    runner = ExperimentRunner(rt, tmp_path)
    try:
        job = runner.submit(spec())
        assert rt.entered.wait(5)
        pending = runner.cancel(job["id"])
        assert pending["state"] == "canceling" and pending["cancel_requested"]
        assert not pending["cleanup_acknowledged"] and not rt.cleaned.is_set()
        rt.release.set()
        stopped = runner.wait(job["id"])
        assert stopped["state"] == "canceled" and stopped["cleanup_acknowledged"] and rt.cleaned.is_set()
        runner.resume(job["id"])
        done = runner.wait(job["id"])
        assert done["state"] == "completed" and done["costs"]["requests"] == 3
        assert done["costs"]["charged_tokens"] == 9
    finally:
        rt.release.set()
        runner.shutdown()


def test_restart_retains_unknown_trial_charge_and_completed_arm(tmp_path):
    class SecondCallFails(FakeRuntime):
        def generate_one(self, *args, **kwargs):
            if len(self.calls) == 1:
                self.calls.append({"failed": True})
                raise RuntimeError("fixture process failure")
            return super().generate_one(*args, **kwargs)

    rt = SecondCallFails()
    runner = ExperimentRunner(rt, tmp_path)
    job = runner.submit(spec(limits={"max_requests": 4, "max_tokens": 512, "max_wall_seconds": 30}))
    first = runner.wait(job["id"])
    assert first["state"] == "failed"
    assert runner.shutdown()
    # Reproduce a process death after persisting a request but before its response.
    db = sqlite3.connect(tmp_path / "research.sqlite3")
    frozen = json.loads(db.execute("SELECT spec FROM research_jobs WHERE id=?", (job["id"],)).fetchone()[0])
    from experiment_schema import canonical, digest
    db.execute("UPDATE research_jobs SET state='running',active_since=?,base_wall=wall_seconds WHERE id=?",
               (time.time(), job["id"]))
    db.execute("UPDATE research_trials SET state='running',finished=NULL,result=NULL,error=NULL WHERE job_id=? AND arm='treatment'",
               (job["id"],))
    db.commit()
    db.close()
    restored = ExperimentRunner(rt, tmp_path)
    try:
        interrupted = restored.status(job["id"])
        assert interrupted["state"] == "interrupted" and not interrupted["cleanup_acknowledged"]
        assert interrupted["costs"]["requests"] == 2 and interrupted["costs"]["charged_tokens"] == 68
        assert len(rt.calls) == 2  # Recovery itself never runs inference.
        restored.resume(job["id"])
        done = restored.wait(job["id"])
        assert done["state"] == "completed" and done["costs"]["requests"] == 3
        assert len(rt.calls) == 3  # Completed baseline was not repeated.
        assert done["costs"]["unknown_cost_requests"] == 1
    finally:
        restored.shutdown()


def test_identity_change_and_concurrent_runner_are_explicit(tmp_path):
    rt = FakeRuntime()
    runner = ExperimentRunner(rt, tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            ExperimentRunner(rt, tmp_path)
        original = rt.generate_one
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            rt.identity["calibration"] = {"changed": True}
            return result
        rt.generate_one = changed
        done = runner.wait(runner.submit(spec())["id"])
        assert done["state"] == "failed" and "identity changed" in done["error"]
        assert len(rt.calls) == 1
        assert done["progress"]["completed_arms"] == 0
    finally:
        runner.shutdown()


@pytest.mark.parametrize("changes", [{"split": "train"}, {"limits": {"max_requests": True}},
                                      {"max_new": True}, {"candidate_frozen": "yes"},
                                      {"extra": 1}, {"items": []}, {"mode": "first_token_only"},
                                      {"split": []}, {"mode": []}, {"eos_policy": {}},
                                      {"treatment": {"pushes": [[0, 3, 10**400, "add"]]}}])
def test_strict_schema_rejects_malformed_inputs(changes):
    with pytest.raises(ValueError):
        experiment_spec(spec(**changes))


def test_exact_grader_rejects_prefix_only_and_extra_explanation():
    assert grade_exact(" 12\n", "12")
    assert not grade_exact("1", "12")
    assert not grade_exact("12 because the rule says so", "12")


def test_runner_rejects_first_token_only_multitoken_results(tmp_path):
    class FirstTokenOnly(FakeRuntime):
        def score_sequence(self, *args, **kwargs):
            result = super().score_sequence(*args, **kwargs)
            result["per_token_logprobs"] = result["per_token_logprobs"][:1]
            result["counts"]["scored_tokens"] = 1
            return result

    runner = ExperimentRunner(FirstTokenOnly(), tmp_path)
    try:
        done = runner.wait(runner.submit(spec(mode="sequence_score"))["id"])
        assert done["state"] == "failed" and "first-token-only" in done["error"]
        assert done["progress"]["completed_arms"] == 0
        assert done["costs"]["requests"] == 1
    finally:
        runner.shutdown()
