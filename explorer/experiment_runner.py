"""Durable, serial, bounded paired execution over an existing Runtime.

The runner does not own model loading, a second model, or candidate discovery.
The Runtime owns admission, its shared lease and request cleanup. Each call is
one arm. Labels are kept in the immutable evaluator spec and never inserted in
generation messages. Only development summaries are available to a proposer.

max_tokens bounds charged input-plus-output tokens, not FLOPs. A context-sized
reservation is persisted before each call; unknown costs retain that charge.
Observed forwards/evaluated tokens and controller process CPU are separate.
"""
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import queue
import sqlite3
import threading
import time
import uuid

from experiment_schema import canonical, digest, experiment_spec, finite_number, grade_exact

TERMINAL = {"completed", "canceled", "deadline", "budget_exhausted", "failed", "interrupted"}
SUCCESS_REASONS = {"completed", "eos", "length"}


class ExperimentRunner:
    def __init__(self, runtime, state_dir, identity=None):
        self.runtime = runtime
        self._identity_override = deepcopy(identity)
        self.identity = deepcopy(identity if identity is not None else self._runtime_identity())
        if not isinstance(self.identity, dict) or not self.identity:
            raise ValueError("A checkpoint/runtime/calibration identity is required")
        canonical(self.identity)
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lease = open(self.state_dir / "research-runner.lock", "a")
        try:
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lease.close()
            raise RuntimeError("An experiment runner already owns this state store") from None
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.state_dir / "research.sqlite3", check_same_thread=False,
                                   timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS research_jobs (
                id TEXT PRIMARY KEY, spec TEXT NOT NULL, spec_hash TEXT NOT NULL,
                identity TEXT NOT NULL, state TEXT NOT NULL, created REAL NOT NULL,
                updated REAL NOT NULL, cancel_requested INTEGER NOT NULL DEFAULT 0,
                wall_seconds REAL NOT NULL DEFAULT 0, process_cpu_seconds REAL NOT NULL DEFAULT 0,
                active_since REAL, base_wall REAL NOT NULL DEFAULT 0, error TEXT);
            CREATE TABLE IF NOT EXISTS research_trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                item_index INTEGER NOT NULL, arm TEXT NOT NULL, state TEXT NOT NULL,
                started REAL NOT NULL, finished REAL, reserved_tokens INTEGER NOT NULL,
                charged_tokens INTEGER NOT NULL, observed_tokens INTEGER,
                wall_seconds REAL NOT NULL DEFAULT 0, process_cpu_seconds REAL NOT NULL DEFAULT 0,
                result TEXT, error TEXT, configuration TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS research_trials_job ON research_trials(job_id, id);
        """)
        self._db.commit()
        self._events = {}
        self._queue = queue.Queue()
        self._stopping = threading.Event()
        self._closed = False
        self._recover()
        self._worker = threading.Thread(target=self._work, name="synapse-research", daemon=True)
        self._worker.start()

    def _runtime_identity(self):
        method = getattr(self.runtime, "research_identity", None)
        if not callable(method):
            if self._identity_override is not None:
                return deepcopy(self._identity_override)
            raise ValueError("Pass identity or provide runtime.research_identity()")
        return method()

    def _recover(self):
        """A process interruption never replays inference automatically."""
        now = time.time()
        with self._lock, self._db:
            for row in self._db.execute("SELECT * FROM research_jobs WHERE state IN ('queued','running','canceling')").fetchall():
                spec = json.loads(row["spec"])
                wall = row["wall_seconds"]
                if row["active_since"] is not None:
                    # Unknown execution/downtime is conservatively charged, bounded by the frozen limit.
                    wall = max(wall, min(spec["limits"]["max_wall_seconds"],
                                         row["base_wall"] + max(0, now - row["active_since"])))
                self._db.execute("UPDATE research_jobs SET state='interrupted', updated=?, wall_seconds=?, active_since=NULL, error=? WHERE id=?",
                                 (now, wall, "Process interrupted; explicit resume required", row["id"]))
                self._db.execute("UPDATE research_trials SET state='interrupted',finished=?,error=? WHERE job_id=? AND state='running'",
                                 (now, "Unknown outcome; full token reservation retained", row["id"]))

    def _job(self, job_id):
        if not isinstance(job_id, str):
            raise ValueError("job_id must be a string")
        row = self._db.execute("SELECT * FROM research_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return row

    def _ensure_open(self):
        if self._closed or self._stopping.is_set():
            raise RuntimeError("Experiment runner is shutting down")

    def submit(self, spec):
        spec = experiment_spec(spec)
        with self._lock, self._db:
            self._ensure_open()
            identity = self._runtime_identity()
            ident = "job-" + uuid.uuid4().hex
            now = time.time()
            self._db.execute("INSERT INTO research_jobs(id,spec,spec_hash,identity,state,created,updated) VALUES(?,?,?,?,?,?,?)",
                             (ident, canonical(spec), digest(spec), canonical(identity), "queued", now, now))
            self._events[ident] = threading.Event()
        self._queue.put(ident)
        return self.status(ident)

    def _trials(self, job_id):
        return self._db.execute("SELECT * FROM research_trials WHERE job_id=? ORDER BY id", (job_id,)).fetchall()

    @staticmethod
    def _successful(trials):
        return {(x["item_index"], x["arm"]): x for x in trials if x["state"] == "completed"}

    @staticmethod
    def _costs(trials, row):
        totals = {"requests": len(trials), "charged_tokens": sum(x["charged_tokens"] for x in trials),
                  "observed_tokens": sum(x["observed_tokens"] or 0 for x in trials),
                  "unknown_cost_requests": sum(x["observed_tokens"] is None for x in trials),
                  "forward_passes": 0, "evaluated_tokens": 0,
                  "wall_seconds": row["wall_seconds"], "process_cpu_seconds": row["process_cpu_seconds"]}
        for trial in trials:
            result = json.loads(trial["result"]) if trial["result"] else {}
            counts = result.get("counts", {})
            for key in ("forward_passes", "evaluated_tokens"):
                v = counts.get(key)
                if type(v) is int and v >= 0:
                    totals[key] += v
        return totals

    def status(self, job_id):
        with self._lock:
            row = self._job(job_id)
            spec = json.loads(row["spec"])
            trials = self._trials(job_id)
            successful = self._successful(trials)
            pairs, summary = [], {"paired_items": 0, "wins": 0, "losses": 0, "ties": 0,
                                  "regressions": 0, "baseline_correct": 0, "treatment_correct": 0}
            for i, item in enumerate(spec["items"]):
                if (i, "baseline") not in successful or (i, "treatment") not in successful:
                    continue
                arms = {arm: json.loads(successful[i, arm]["result"]) for arm in ("baseline", "treatment")}
                pair = {"item_id": item["id"], "baseline": arms["baseline"], "treatment": arms["treatment"]}
                if spec["mode"] == "generation":
                    a, b = (grade_exact(arms[arm]["text"], item["answer"]) for arm in ("baseline", "treatment"))
                    pair.update(baseline_correct=a, treatment_correct=b)
                    summary["baseline_correct"] += int(a)
                    summary["treatment_correct"] += int(b)
                    outcome = "wins" if b and not a else "losses" if a and not b else "ties"
                    summary[outcome] += 1
                    summary["regressions"] += int(a and not b)
                else:
                    pair["delta_total_logprob"] = arms["treatment"]["total_logprob"] - arms["baseline"]["total_logprob"]
                pairs.append(pair)
            summary["paired_items"] = len(pairs)
            if spec["mode"] == "sequence_score":
                summary = {"paired_items": len(pairs), "metric": "complete_sequence_logprob",
                           "mean_delta_total_logprob": (sum(p["delta_total_logprob"] for p in pairs) / len(pairs)) if pairs else None,
                           "accuracy_measured": False}
            else:
                summary["metric"] = "exact_answer_accuracy"
            return {"id": row["id"], "state": row["state"], "name": spec["name"],
                    "mode": spec["mode"], "split": spec["split"], "seed": spec["seed"],
                    "spec_hash": row["spec_hash"], "identity": json.loads(row["identity"]),
                    "configurations": {arm: spec[arm] for arm in ("baseline", "treatment")},
                    "limits": spec["limits"], "progress": {"completed_arms": len(successful),
                    "total_arms": 2 * len(spec["items"]), "paired_items": len(pairs)},
                    "cancel_requested": bool(row["cancel_requested"]),
                    "cleanup_acknowledged": row["state"] in TERMINAL - {"interrupted"} and all(
                        json.loads(t["result"] or "{}").get("cleanup") == {
                            "interventions_cleared": True, "cache_cleared": True} for t in trials),
                    "costs": self._costs(trials, row), "outcomes": summary, "pairs": pairs,
                    "trials": [{"attempt": x["id"], "item_id": spec["items"][x["item_index"]]["id"],
                                "arm": x["arm"], "state": x["state"], "charged_tokens": x["charged_tokens"],
                                "observed_tokens": x["observed_tokens"], "result": json.loads(x["result"]) if x["result"] else None,
                                "error": x["error"]} for x in trials], "error": row["error"],
                    "created_at": row["created"], "updated_at": row["updated"],
                    "accounting_policy": "context-sized reservation; unknown work charged conservatively; process CPU includes other process threads"}

    def list(self, *, limit=100):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Job list limit must be 1..1000")
        with self._lock:
            rows = self._db.execute(
                "SELECT id,state,spec,created,updated FROM research_jobs ORDER BY created DESC,id LIMIT ?",
                (limit,)).fetchall()
            return [{"id": row["id"], "state": row["state"],
                     "name": json.loads(row["spec"])["name"],
                     "split": json.loads(row["spec"])["split"],
                     "created_at": row["created"], "updated_at": row["updated"]} for row in rows]

    def feedback(self, job_id):
        """The only proposer-facing reader: no labels, items, validation or final data."""
        report = self.status(job_id)
        if report["split"] != "development":
            raise ValueError("Only development feedback may be supplied to a researcher")
        return {k: report[k] for k in ("id", "state", "split", "spec_hash", "identity", "outcomes", "costs")}

    def cancel(self, job_id):
        with self._lock, self._db:
            row = self._job(job_id)
            if row["state"] not in TERMINAL:
                self._events.setdefault(job_id, threading.Event()).set()
                state = "canceled" if row["state"] == "queued" else "canceling"
                self._db.execute("UPDATE research_jobs SET state=?,cancel_requested=1,updated=? WHERE id=?",
                                 (state, time.time(), job_id))
        return self.status(job_id)

    def resume(self, job_id):
        with self._lock, self._db:
            self._ensure_open()
            row = self._job(job_id)
            if row["state"] not in {"canceled", "interrupted", "failed"}:
                raise ValueError("Only a canceled, interrupted or failed job may resume")
            if canonical(json.loads(row["identity"])) != canonical(self._runtime_identity()):
                raise ValueError("Checkpoint/runtime/calibration identity differs; replay is incompatible")
            if digest(json.loads(row["spec"])) != row["spec_hash"]:
                raise ValueError("Stored experiment spec changed; resume refused")
            self._events[job_id] = threading.Event()
            self._db.execute("UPDATE research_jobs SET state='queued',cancel_requested=0,error=NULL,updated=? WHERE id=?",
                             (time.time(), job_id))
        self._queue.put(job_id)
        return self.status(job_id)

    def _reservation(self):
        caps = self.runtime.capabilities()
        size = caps.get("context_tokens", 2048)
        if type(size) is not int or not 1 <= size <= 1_048_576:
            raise ValueError("Runtime must advertise a bounded context_tokens value")
        return size

    @staticmethod
    def _observed(result):
        if not isinstance(result, dict):
            return None
        counts = result.get("counts", {})
        prompt = counts.get("prompt_tokens")
        values = [counts[k] for k in ("output_tokens", "scored_tokens") if k in counts]
        if type(prompt) is not int or not values or any(type(v) is not int or v < 0 for v in values):
            return None
        output = max(values)
        if prompt < 0:
            return None
        return prompt + output

    @staticmethod
    def _validate_complete(result, mode):
        counts = result.get("counts", {})
        if ExperimentRunner._observed(result) is None:
            raise ValueError("Runtime must report exact prompt and output/scored token counts")
        if mode == "generation":
            ids = result.get("output_token_ids")
            if not isinstance(result.get("text"), str) or type(ids) is not list or any(type(x) is not int or x < 0 for x in ids):
                raise ValueError("Generation result needs exact text and output token ids")
            if counts.get("output_tokens") != len(ids):
                raise ValueError("Generation token count differs from exact output ids")
        else:
            ids, scores = result.get("completion_token_ids"), result.get("per_token_logprobs")
            if (type(ids) is not list or not ids or any(type(x) is not int or x < 0 for x in ids)
                    or type(scores) is not list or len(scores) != len(ids)
                    or any(not finite_number(x) or x > 1e-8 for x in scores)
                    or counts.get("scored_tokens") != len(ids)):
                raise ValueError("Complete scoring requires every contextual completion token; first-token-only results are invalid")
            total = result.get("total_logprob")
            mean = result.get("mean_logprob")
            if not finite_number(total) or abs(total - sum(scores)) > 1e-8:
                raise ValueError("Complete-sequence total does not match per-token scores")
            if not finite_number(mean) or abs(mean - total / len(ids)) > 1e-8:
                raise ValueError("Complete-sequence mean does not match per-token scores")

    def _execute(self, job_id):
        with self._lock, self._db:
            row = self._job(job_id)
            if row["state"] != "queued":
                return
            spec = json.loads(row["spec"])
            frozen_identity = json.loads(row["identity"])
            base_wall = row["wall_seconds"]
            self._db.execute("UPDATE research_jobs SET state='running',active_since=?,base_wall=?,updated=? WHERE id=?",
                             (time.time(), base_wall, time.time(), job_id))
        event = self._events[job_id]
        start, cpu_start = time.monotonic(), time.process_time()
        deadline = start + max(0, spec["limits"]["max_wall_seconds"] - base_wall)
        state, error = "completed", None
        try:
            for i, item in enumerate(spec["items"]):
                # Alternating order is frozen by item index and seed, avoiding a fixed first-arm advantage.
                order = ("baseline", "treatment") if (i + spec["seed"]) % 2 == 0 else ("treatment", "baseline")
                for arm in order:
                    with self._lock:
                        trials = self._trials(job_id)
                        if (i, arm) in self._successful(trials):
                            continue
                    if event.is_set() or self._stopping.is_set():
                        state = "canceled"
                        return
                    if time.monotonic() >= deadline:
                        state = "deadline"
                        return
                    if canonical(self._runtime_identity()) != canonical(frozen_identity):
                        raise ValueError("Runtime identity changed during experiment; arm refused")
                    reserve = self._reservation()
                    with self._lock, self._db:
                        trials = self._trials(job_id)
                        charged = sum(t["charged_tokens"] for t in trials)
                        if len(trials) >= spec["limits"]["max_requests"] or charged + reserve > spec["limits"]["max_tokens"]:
                            state = "budget_exhausted"
                            return
                        cur = self._db.execute("INSERT INTO research_trials(job_id,item_index,arm,state,started,reserved_tokens,charged_tokens,configuration) VALUES(?,?,?,'running',?,?,?,?)",
                                               (job_id, i, arm, time.time(), reserve, reserve, canonical(spec[arm])))
                        trial_id = cur.lastrowid
                    call_start, call_cpu = time.monotonic(), time.process_time()
                    result, failure, trial_state = None, None, "failed"
                    try:
                        kwargs = {"pushes": deepcopy(spec[arm]["pushes"]), "cancel": event, "deadline": deadline}
                        if spec["mode"] == "generation":
                            result = self.runtime.generate_one([{"role": "user", "content": item["prompt"]}],
                                                               max_new=spec["max_new"], **kwargs)
                        else:
                            result = self.runtime.score_sequence(item["prompt"], item["answer"],
                                                                 chat=spec["chat"], prefix=spec["prefix"],
                                                                 eos_policy=spec["eos_policy"], **kwargs)
                        if type(result) is not dict:
                            raise ValueError("Runtime result must be a JSON object")
                        canonical(result)
                        if canonical(self._runtime_identity()) != canonical(frozen_identity):
                            raise ValueError("Runtime identity changed during inference; result refused")
                        if "identity" in result and canonical(result["identity"]) != canonical(frozen_identity):
                            raise ValueError("Result identity differs from the frozen experiment")
                        completed = result.get("completed", result.get("complete", False))
                        reason = result.get("finish_reason")
                        if completed is True and reason in SUCCESS_REASONS:
                            self._validate_complete(result, spec["mode"])
                            trial_state = "completed"
                        elif reason in {"canceled", "deadline"}:
                            trial_state = reason
                        else:
                            raise ValueError("Runtime returned an incomplete or unsupported outcome")
                    except Exception as exc:
                        failure = f"{type(exc).__name__}: {exc}"
                        try:
                            if not isinstance(result, dict):
                                result = None
                            canonical(result)
                        except (TypeError, ValueError):
                            result = None
                        costs = getattr(exc, "research_costs", None)
                        cleanup = getattr(exc, "research_cleanup", None)
                        if isinstance(costs, dict) or isinstance(cleanup, dict):
                            result = {"counts": costs or {}, "completed": False,
                                      "finish_reason": "error", "cleanup": cleanup}
                    finally:
                        elapsed, cpu = time.monotonic() - call_start, time.process_time() - call_cpu
                        observed = self._observed(result or {})
                        # An invalid over-bound result is retained as evidence, never counted as a successful arm.
                        if observed is not None and observed > reserve:
                            failure = "Runtime exceeded its advertised context token bound"
                            trial_state = "failed"
                        charged = observed if observed is not None else reserve
                        with self._lock, self._db:
                            self._db.execute("UPDATE research_trials SET state=?,finished=?,charged_tokens=?,observed_tokens=?,wall_seconds=?,process_cpu_seconds=?,result=?,error=? WHERE id=?",
                                             (trial_state, time.time(), charged, observed, elapsed, cpu,
                                              canonical(result) if result is not None else None, failure, trial_id))
                            self._db.execute("UPDATE research_jobs SET wall_seconds=?,updated=? WHERE id=?",
                                             (base_wall + time.monotonic() - start, time.time(), job_id))
                    if failure:
                        raise RuntimeError(failure)
                    if trial_state != "completed":
                        state = trial_state
                        return
                    if event.is_set():
                        state = "canceled"
                        return
                    if time.monotonic() >= deadline:
                        state = "deadline"
                        return
        except Exception as exc:
            state, error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock, self._db:
                self._db.execute("UPDATE research_jobs SET state=?,wall_seconds=?,process_cpu_seconds=process_cpu_seconds+?,active_since=NULL,error=?,updated=? WHERE id=?",
                                 (state, base_wall + time.monotonic() - start,
                                  time.process_time() - cpu_start, error, time.time(), job_id))

    def _work(self):
        while True:
            ident = self._queue.get()
            try:
                if ident is None:
                    return
                self._execute(ident)
            finally:
                self._queue.task_done()

    def wait(self, job_id, timeout=30):
        if not finite_number(timeout) or timeout <= 0:
            raise ValueError("wait timeout must be positive and finite")
        end = time.monotonic() + timeout
        while True:
            report = self.status(job_id)
            if report["state"] in TERMINAL:
                return report
            if time.monotonic() >= end:
                raise TimeoutError(f"Job {job_id} has not finished")
            self._stopping.wait(min(.01, max(0, end - time.monotonic())))

    def shutdown(self, timeout=30):
        """Request cancellation and wait for actual runtime cleanup before closing."""
        if self._closed:
            return True
        self._stopping.set()
        with self._lock:
            for event in self._events.values():
                event.set()
        self._queue.put(None)
        self._worker.join(timeout)
        if self._worker.is_alive():
            return False
        with self._lock:
            self._db.close()
            self._lease.close()
            self._closed = True
        return True
