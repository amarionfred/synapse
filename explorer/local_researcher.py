"""Bounded selection from a frozen action menu by the inspected local model.

This is a synchronous client of the existing runtime, not another model or
worker. Its only feedback channel accepts aggregate development outcomes.
Untouched evaluation belongs to the experiment runner, outside this object.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import random
import threading
import time

from presets import canonical, normalize_config, normalize_pushes


class ResearchLimit(RuntimeError):
    """A prospective budget or stop condition prevented another request."""


class LocalResearcher:
    """Choose semantic actions using complete contextual display-ID likelihoods.

    ``record(event)`` can append each event to the runner's existing SQLite
    journal. It is called before inference as well as afterward. Failed calls
    retain their conservative token reservation when actual cost is unknown.
    ``max_tokens`` bounds charged *evaluated* tokens, including prompt forwards,
    rather than counting only returned answer tokens.
    """

    def __init__(self, runtime, researcher_config, target_config, budget, record=None, state=None):
        self.runtime = runtime
        self.researcher_config = normalize_config(researcher_config)
        self.target_config = normalize_config(target_config)
        if not hasattr(runtime, "lock"):
            raise ValueError("Local research requires the shared runtime lock")
        if not isinstance(budget, dict) or set(budget) != {"max_requests", "max_tokens", "max_seconds", "max_decisions"}:
            raise ValueError("Budget requires max_requests, max_tokens, max_seconds and max_decisions")
        for name, maximum in (("max_requests", 4096), ("max_tokens", 2**40), ("max_decisions", 128)):
            if type(budget[name]) is not int or not 1 <= budget[name] <= maximum:
                raise ValueError("Invalid bounded research budget: " + name)
        if (type(budget["max_seconds"]) not in (float, int) or
                not math.isfinite(budget["max_seconds"]) or not 0 < budget["max_seconds"] <= 3600):
            raise ValueError("max_seconds must be finite and in (0, 3600]")
        self.budget = canonical(budget)
        self.record = record
        self._state_lock = threading.RLock()
        self.started = time.monotonic()
        self._deadline = self.started + self.budget["max_seconds"]
        self._frozen = None
        self._menu = {}
        self._decisions = 0
        self._events = []
        self._cost = {"requests": 0, "researcher_requests": 0, "target_requests": 0,
                      "charged_tokens": 0, "actual_tokens": 0, "unknown_cost_requests": 0,
                      "forward_passes": 0, "failed_requests": 0,
                      "cpu_seconds": 0.0, "request_wall_seconds": 0.0}
        if state is not None:
            self._restore_state(state)

    def _restore_state(self, state):
        """Restore the same protocol from runner-held state, retaining all charges.

        A persisted request_started without request_finished is charged at its
        full reservation. It must never be silently rerun as unspent work.
        """
        encoded = json.dumps(state, allow_nan=False)
        if len(encoded.encode()) > 16 * 1024 * 1024:
            raise ValueError("Researcher journal exceeds 16 MiB")
        state = json.loads(encoded)
        if (not isinstance(state, dict) or state.get("budget") != self.budget or
                state.get("researcher_config") != self.researcher_config or
                state.get("target_config") != self.target_config or
                state.get("kind") != "inspected-local-model"):
            raise ValueError("Researcher resume requires the same frozen configuration and budget")
        events = state.get("events")
        if not isinstance(events, list) or len(events) > 20000:
            raise ValueError("Invalid researcher journal")
        starts, finishes = {}, {}
        decisions = 0
        for index, event in enumerate(events):
            if not isinstance(event, dict) or event.get("index") != index:
                raise ValueError("Researcher journal order is invalid")
            if event.get("event") == "request_started":
                request_id, reserve = event.get("request_id"), event.get("token_reservation")
                if (type(request_id) is not int or request_id != len(starts) + 1 or
                        type(reserve) is not int or not 1 <= reserve <= 2**40 or
                        event.get("role") not in ("researcher", "target")):
                    raise ValueError("Invalid researcher request reservation")
                starts[request_id] = event
            elif event.get("event") == "request_finished":
                request_id = event.get("request_id")
                if request_id not in starts or request_id in finishes:
                    raise ValueError("Invalid researcher request completion")
                actual = event.get("actual_evaluated_tokens")
                if actual is not None and (type(actual) is not int or actual < 0):
                    raise ValueError("Invalid researcher actual cost")
                finishes[request_id] = event
            elif event.get("event") == "decision_started":
                decisions += 1
                if event.get("decision") != decisions:
                    raise ValueError("Invalid researcher decision journal")
            elif event.get("event") == "decision" and event.get("decision") != decisions:
                raise ValueError("Invalid researcher decision completion")
        # Reconstruct bounded counters from journal facts, never trust a caller's
        # editable summary to restore spent tokens or failed attempts to zero.
        for request_id, event in starts.items():
            self._cost["requests"] += 1
            self._cost[event["role"] + "_requests"] += 1
            finished = finishes.get(request_id)
            actual = finished.get("actual_evaluated_tokens") if finished else None
            self._cost["charged_tokens"] += event["token_reservation"] if actual is None else actual
            if actual is None:
                self._cost["unknown_cost_requests"] += 1
            else:
                self._cost["actual_tokens"] += actual
                result = finished.get("result")
                count = result.get("counts", {}).get("forward_passes", 0) if isinstance(result, dict) else 0
                if type(count) is not int or count < 0:
                    raise ValueError("Invalid researcher forward count")
                self._cost["forward_passes"] += count
            if finished is None or finished.get("error"):
                self._cost["failed_requests"] += 1
        summary = state.get("cost", {})
        for name in ("cpu_seconds", "request_wall_seconds"):
            value = summary.get(name, 0.0)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid researcher elapsed cost")
            self._cost[name] = value
        elapsed = state.get("elapsed_seconds")
        if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid researcher elapsed budget")
        self.started -= elapsed
        self._deadline = self.started + self.budget["max_seconds"]
        self._events, self._decisions, self._frozen = events, decisions, state.get("frozen_protocol")
        if self._frozen is not None and (not isinstance(self._frozen, str) or len(self._frozen) != 64):
            raise ValueError("Invalid frozen researcher protocol")

    def snapshot(self):
        with self._state_lock:
            # Whole episode state may exceed one request's 256 KiB bound. It is
            # already assembled exclusively from bounded immutable JSON events.
            return json.loads(json.dumps({"kind": "inspected-local-model", "external_agent": False,
                              "researcher_config": self.researcher_config, "target_config": self.target_config,
                              "budget": self.budget, "cost": self._cost, "decisions": self._decisions,
                              "elapsed_seconds": time.monotonic() - self.started,
                              "frozen_protocol": self._frozen, "events": self._events}, allow_nan=False))

    def _emit(self, event):
        event = canonical({"index": len(self._events), **event})
        self._events.append(event)
        if self.record is not None:
            self.record(canonical(event))

    def _stop(self, cancel, deadline):
        if cancel is not None and cancel.is_set():
            return "canceled"
        if deadline is not None and time.monotonic() >= deadline:
            return "deadline"
        if time.monotonic() >= self._deadline:
            return "budget"
        return None

    @contextmanager
    def _lease(self, cancel, deadline):
        # Poll the ordinary shared lock only while queued, so a blocked legacy
        # request cannot make this client's own deadline silently disappear.
        acquired = False
        while not acquired:
            reason = self._stop(cancel, deadline)
            if reason:
                raise ResearchLimit(reason)
            acquired = self.runtime.lock.acquire(timeout=0.05)
        try:
            yield
        finally:
            self.runtime.lock.release()

    def _context_limit(self):
        caps = self.runtime.capabilities()
        limit = caps.get("context_tokens", 2048)
        if type(limit) is not int or not 1 <= limit <= 1048576:
            raise ValueError("Runtime must report a bounded context_tokens value")
        return limit

    def _request(self, role, operation, request, *, config, cancel, deadline):
        reason = self._stop(cancel, deadline)
        if reason:
            raise ResearchLimit(reason)
        context = self._context_limit()
        # The score contract allows at most context completion tokens, with
        # full-prefix evaluation as a legal backend. context^2 bounds that work.
        # A backend may expose a tighter audited preflight bound without inference.
        bound = context * context if operation == "score" else context * (request["max_new"] + 1)
        preflight = getattr(self.runtime, "research_token_bound", None)
        if preflight is not None:
            bound = preflight(operation, **request)
        if type(bound) is not int or not 1 <= bound <= 2**40:
            raise ValueError("Invalid runtime token-cost bound")
        if self._cost["requests"] >= self.budget["max_requests"]:
            raise ResearchLimit("request_budget")
        if self._cost["charged_tokens"] + bound > self.budget["max_tokens"]:
            raise ResearchLimit("token_budget")
        self._cost["requests"] += 1
        self._cost[role + "_requests"] += 1
        self._cost["charged_tokens"] += bound
        request_id = self._cost["requests"]
        started, cpu_started = time.monotonic(), time.process_time()
        # Persist the reservation before entering native inference. A crash or
        # malformed/failed result cannot erase attempted work from the budget.
        self._emit({"event": "request_started", "request_id": request_id, "role": role,
                    "operation": operation, "input": request, "token_reservation": bound,
                    "researcher_config": self.researcher_config,
                    "target_config": self.target_config if role == "researcher" else config,
                    "applied_config": config,
                    "cost": self._cost})
        result, error, actual = None, None, None
        try:
            method = self.runtime.score_sequence if operation == "score" else self.runtime.generate_one
            result = method(**request, pushes=config["pushes"], cancel=cancel,
                            deadline=min(self._deadline, deadline) if deadline is not None else self._deadline)
            if not isinstance(result, dict) or not isinstance(result.get("counts"), dict):
                raise ValueError("Runtime omitted actual request accounting")
            counts = result["counts"]
            actual = counts.get("evaluated_tokens")
            forwards = counts.get("forward_passes")
            if type(actual) is not int or actual < 0 or type(forwards) is not int or forwards < 0:
                actual = None
                raise ValueError("Runtime returned invalid token/forward accounting")
            self._cost["actual_tokens"] += actual
            self._cost["forward_passes"] += forwards
            self._cost["charged_tokens"] += actual - bound
            if actual > bound:
                raise ValueError("Runtime exceeded its prospective evaluated-token bound")
            return result
        except Exception as exc:
            error = str(exc)[:1000]
            self._cost["failed_requests"] += 1
            if actual is None:
                self._cost["unknown_cost_requests"] += 1
            raise
        finally:
            self._cost["cpu_seconds"] += time.process_time() - cpu_started
            self._cost["request_wall_seconds"] += time.monotonic() - started
            self._emit({"event": "request_finished", "request_id": request_id, "role": role,
                        "operation": operation, "result": result, "error": error,
                        "actual_evaluated_tokens": actual, "cost": self._cost})

    @staticmethod
    def _feedback(feedback, action_ids):
        if not isinstance(feedback, (tuple, list)) or len(feedback) > 128:
            raise ValueError("Feedback must be at most 128 development summaries")
        normalized = []
        for item in feedback:
            if (not isinstance(item, dict) or set(item) != {"split", "action_id", "correct", "total"} or
                    item["split"] != "development" or item["action_id"] not in action_ids or
                    type(item["correct"]) is not int or type(item["total"]) is not int or
                    not 0 <= item["correct"] <= item["total"] <= 100000):
                raise ValueError("Only aggregate development feedback for this fixed menu is permitted")
            normalized.append(canonical(item))
        return normalized

    @staticmethod
    def _actions(menu):
        if not isinstance(menu, (tuple, list)) or not 1 <= len(menu) <= 32:
            raise ValueError("The fixed action menu needs 1..32 actions")
        out = {}
        for action in menu:
            if (not isinstance(action, dict) or set(action) != {"id", "description", "pushes"} or
                    not isinstance(action["id"], str) or not 1 <= len(action["id"]) <= 64 or
                    not isinstance(action["description"], str) or len(action["description"]) > 1000 or
                    action["id"] in out):
                raise ValueError("Actions require distinct semantic IDs, fixed descriptions and pushes")
            out[action["id"]] = {"id": action["id"], "description": action["description"],
                                  "pushes": normalize_pushes(action["pushes"])}
        return dict(sorted(out.items()))

    def choose(self, menu, *, brief="", feedback=(), seed=0, display_ids=None, cancel=None, deadline=None):
        with self._state_lock:
            return self._choose(menu, brief=brief, feedback=feedback, seed=seed, display_ids=display_ids,
                                cancel=cancel, deadline=deadline)

    def _choose(self, menu, *, brief, feedback, seed, display_ids, cancel, deadline):
        """Score every displayed action ID completely; never select a partial result.

        Menu, brief, display vocabulary and seed freeze on the first valid
        decision. Subsequent adaptation is restricted to development summaries.
        Ties resolve by semantic ID, independently of presentation positions.
        """
        if self._decisions >= self.budget["max_decisions"]:
            raise ResearchLimit("decision_budget")
        self._decisions += 1
        self._emit({"event": "decision_started", "decision": self._decisions})
        scores, presentation = [], []
        try:
            actions = self._actions(menu)
            if not isinstance(brief, str) or len(brief) > 8000 or type(seed) is not int or not 0 <= seed < 2**63:
                raise ValueError("Provide a bounded frozen brief and nonnegative integer seed")
            if deadline is not None and (type(deadline) not in (float, int) or not math.isfinite(deadline)):
                raise ValueError("Deadline must be an absolute finite monotonic time")
            displays = [str(i + 1) for i in range(len(actions))] if display_ids is None else display_ids
            if (not isinstance(displays, (tuple, list)) or len(displays) != len(actions) or
                    any(not isinstance(d, str) or not 1 <= len(d) <= 16 or not d.isalnum() for d in displays) or
                    len(set(displays)) != len(displays)):
                raise ValueError("Display IDs must be distinct bounded alphanumeric strings")
            protocol = canonical({"menu": list(actions.values()), "brief": brief,
                                  "display_ids": list(displays), "seed": seed})
            digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
            if self._frozen is not None and self._frozen != digest:
                raise ValueError("Measured researcher protocol is frozen; external adaptive hints are not permitted")
            self._frozen, self._menu = digest, actions
            development = self._feedback(feedback, actions)
            rng = random.Random(seed + self._decisions - 1)
            order, labels = list(actions), list(displays)
            rng.shuffle(order)
            rng.shuffle(labels)
            presentation = [{"display_id": display, "action_id": action_id,
                             "description": actions[action_id]["description"],
                             "pushes": actions[action_id]["pushes"]}
                            for display, action_id in zip(labels, order)]
            prompt = ("Choose one action for the fixed target. Return its displayed ID only.\n"
                      + brief + "\nTarget configuration: " + json.dumps(self.target_config, sort_keys=True)
                      + "\nAction menu: " + json.dumps(presentation, sort_keys=True)
                      + "\nDevelopment feedback: " + json.dumps(development, sort_keys=True) + "\nAction ID:")
            with self._lease(cancel, deadline):
                for item in presentation:
                    result = self._request("researcher", "score", {"text": prompt,
                                           "completion": " " + item["display_id"], "chat": True,
                                           "prefix": "", "eos_policy": "exclude"},
                                           config=self.researcher_config, cancel=cancel, deadline=deadline)
                    if not result.get("completed"):
                        raise ResearchLimit(result.get("finish_reason", "incomplete"))
                    token_ids, logprobs = result.get("completion_token_ids"), result.get("per_token_logprobs")
                    if (result.get("first_token_only") or not isinstance(token_ids, list) or not token_ids or
                            not isinstance(logprobs, list) or len(logprobs) != len(token_ids) or
                            any(type(x) not in (float, int) or not math.isfinite(x) or x > 1e-6 for x in logprobs)):
                        raise ValueError("Menu actions require complete finite per-token contextual scores")
                    scores.append({"action_id": item["action_id"], "display_id": item["display_id"],
                                   "completion_token_ids": token_ids, "per_token_logprobs": logprobs,
                                   "total_logprob": sum(logprobs), "identity": result.get("identity")})
            selected = min(scores, key=lambda row: (-row["total_logprob"], row["action_id"]))
            decision = {"completed": True, "finish_reason": "completed", "action_id": selected["action_id"],
                        "action": actions[selected["action_id"]], "presentation": presentation, "scores": scores,
                        "researcher_config": self.researcher_config, "target_config": self.target_config,
                        "cost": self._cost, "decision": self._decisions}
        except Exception as exc:
            decision = {"completed": False, "finish_reason": str(exc) if isinstance(exc, ResearchLimit) else "failed",
                        "error": str(exc)[:1000], "action_id": None, "presentation": presentation,
                        "scores": scores, "cost": self._cost, "decision": self._decisions}
        self._emit({"event": "decision", **decision})
        return canonical(decision)

    def generate_target(self, action_id, messages, *, max_new=32, split="development", cancel=None, deadline=None):
        with self._state_lock:
            return self._generate_target(action_id, messages, max_new=max_new, split=split,
                                         cancel=cancel, deadline=deadline)

    def _generate_target(self, action_id, messages, *, max_new, split, cancel, deadline):
        """Run a known action against a development prompt without exposing labels.

        Validation/final generation is deliberately left to the runner and never
        becomes a feedback channel to this measured proposer object.
        """
        if split != "development" or action_id not in self._menu:
            raise ValueError("Target requests require a known development action")
        if type(max_new) is not int or not 1 <= max_new <= 256:
            raise ValueError("Target generation needs 1..256 output tokens")
        if (not isinstance(messages, list) or not 1 <= len(messages) <= 64 or
                any(not isinstance(m, dict) or set(m) != {"role", "content"} or
                    m["role"] not in ("user", "system", "assistant") or
                    not isinstance(m["content"], str) for m in messages) or
                sum(len(m["content"]) for m in messages) > 32768):
            raise ValueError("Provide bounded ordinary messages without evaluation labels")
        config = {"pushes": normalize_pushes(self.target_config["pushes"] + self._menu[action_id]["pushes"])}
        with self._lease(cancel, deadline):
            return self._request("target", "generate", {"messages": canonical(messages), "max_new": max_new},
                                 config=config, cancel=cancel, deadline=deadline)


class ResearcherStore:
    """Durable local episodes in the existing research DB; owns no model/worker.

    Every reservation is committed before inference. The single operation lock
    prevents concurrent updates, while status readers can see in-flight progress.
    Only a frozen menu/brief and development aggregates reach the proposer.
    """
    def __init__(self, runtime, state_dir):
        from pathlib import Path
        import sqlite3
        self.runtime = runtime
        Path(state_dir).mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(Path(state_dir) / "research.sqlite3", check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""CREATE TABLE IF NOT EXISTS research_episodes (
            id TEXT PRIMARY KEY, protocol TEXT NOT NULL, identity TEXT NOT NULL,
            snapshot TEXT NOT NULL, state TEXT NOT NULL, updated REAL NOT NULL)""")
        self._db.execute("UPDATE research_episodes SET state='interrupted' WHERE state='running'")
        self._db.commit()
        self._db_lock = threading.RLock()
        self._operation = threading.Lock()

    def create(self, protocol):
        import uuid
        if not isinstance(protocol, dict) or set(protocol) - {
                "menu", "brief", "seed", "display_ids", "researcher_config", "target_config", "budget"}:
            raise ValueError("Invalid fixed researcher protocol")
        protocol = canonical(protocol)
        local = LocalResearcher(self.runtime, protocol.get("researcher_config", {"pushes": []}),
                                protocol.get("target_config", {"pushes": []}), protocol.get("budget"))
        local._actions(protocol.get("menu"))
        brief, seed = protocol.get("brief", ""), protocol.get("seed", 0)
        if not isinstance(brief, str) or len(brief) > 8000 or type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Provide a bounded frozen brief and nonnegative integer seed")
        displays = protocol.get("display_ids")
        if displays is not None and (not isinstance(displays, list) or len(displays) != len(protocol["menu"]) or
                any(not isinstance(d, str) or not 1 <= len(d) <= 16 or not d.isalnum() for d in displays) or
                len(set(displays)) != len(displays)):
            raise ValueError("Invalid frozen display IDs")
        ident = "researcher-" + uuid.uuid4().hex
        identity = canonical(self.runtime.research_identity())
        with self._db_lock, self._db:
            self._db.execute("INSERT INTO research_episodes VALUES(?,?,?,?,?,?)",
                             (ident, json.dumps(protocol), json.dumps(identity),
                              json.dumps(local.snapshot()), "ready", time.time()))
        return self.get(ident)

    def get(self, ident):
        if not isinstance(ident, str):
            raise ValueError("Researcher ID must be a string")
        with self._db_lock:
            row = self._db.execute("SELECT protocol,identity,snapshot,state,updated FROM research_episodes WHERE id=?",
                                   (ident,)).fetchone()
        if row is None:
            raise KeyError(ident)
        return {"id": ident, "protocol": json.loads(row[0]), "identity": json.loads(row[1]),
                "snapshot": json.loads(row[2]), "state": row[3], "updated_at": row[4]}

    def _save(self, ident, local, state):
        snapshot = json.dumps(local.snapshot(), allow_nan=False)
        if len(snapshot.encode()) > 16 * 1024 * 1024:
            raise ValueError("Researcher journal exceeds 16 MiB")
        with self._db_lock, self._db:
            self._db.execute("UPDATE research_episodes SET snapshot=?,state=?,updated=? WHERE id=?",
                             (snapshot, state, time.time(), ident))

    def run(self, ident, feedback=(), *, cancel=None, deadline=None):
        if not self._operation.acquire(blocking=False):
            raise RuntimeError("Another local researcher decision is running")
        local = None
        try:
            episode = self.get(ident)
            if episode["identity"] != self.runtime.research_identity():
                raise ValueError("Researcher checkpoint/runtime/calibration identity changed")
            protocol, snapshot = episode["protocol"], episode["snapshot"]
            # Idle and interrupted time conservatively consume this episode's
            # wall budget. A process restart cannot reset it.
            snapshot["elapsed_seconds"] += max(0., time.time() - episode["updated_at"])
            local = LocalResearcher(self.runtime, snapshot["researcher_config"], snapshot["target_config"],
                                    snapshot["budget"], state=snapshot)
            local.record = lambda event: self._save(ident, local, "running")
            result = local.choose(protocol["menu"], brief=protocol.get("brief", ""), feedback=feedback,
                                  seed=protocol.get("seed", 0), display_ids=protocol.get("display_ids"),
                                  cancel=cancel, deadline=deadline)
            if episode["identity"] != self.runtime.research_identity():
                raise ValueError("Researcher identity changed during inference")
            return {"id": ident, **result}
        finally:
            try:
                if local is not None:
                    self._save(ident, local, "ready")
            finally:
                self._operation.release()

    def close(self):
        with self._db_lock:
            self._db.close()
