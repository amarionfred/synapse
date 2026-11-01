"""Immutable, identity-bound activation configurations in the research database.

Replay/restore return a request configuration. They never install persistent
hooks, change weights, or claim that a stored evaluation reproduces elsewhere.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import time


def canonical(value):
    """A bounded JSON copy; reject nonfinite values and non-JSON Python objects."""
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Metadata must be finite JSON") from exc
    if len(encoded.encode("utf-8")) > 262144:
        raise ValueError("Research metadata exceeds 256 KiB")
    return json.loads(encoded)


def normalize_identity(identity):
    value = canonical(identity)
    if (not isinstance(value, dict) or
            not all(key in value for key in ("checkpoint", "runtime", "calibration")) or
            not isinstance(value["checkpoint"], dict) or not value["checkpoint"] or
            not isinstance(value["runtime"], dict) or not value["runtime"] or
            not isinstance(value["calibration"], dict)):
        raise ValueError("Identity requires checkpoint, runtime and explicit calibration objects")
    return value


def normalize_pushes(pushes):
    """Normalize legacy (layer, neuron, dose, mode) tuples without composition ambiguity."""
    if not isinstance(pushes, (list, tuple)) or len(pushes) > 64:
        raise ValueError("A configuration needs at most 64 MLP interventions")
    out = []
    for push in pushes:
        if not isinstance(push, (list, tuple)) or len(push) != 4:
            raise ValueError("Interventions use [layer, neuron, dose, mode]")
        layer, neuron, dose, mode = push
        if type(layer) is not int or not 0 <= layer < 1024:
            raise ValueError("Invalid intervention layer")
        if neuron is not None and (type(neuron) is not int or not 0 <= neuron < 1000000):
            raise ValueError("Invalid intervention neuron")
        if (type(dose) not in (int, float) or not math.isfinite(dose) or abs(dose) > 1000 or
                mode not in ("add", "mul") or (neuron is None and mode != "mul")):
            raise ValueError("Use a finite bounded dose and mode add or mul; whole layers multiply")
        for old in out:
            if old[0] == layer and (old[1] is None or neuron is None or old[1] == neuron):
                raise ValueError("Overlapping interventions have ambiguous composition")
        out.append([layer, neuron, float(dose), mode])
    return sorted(out, key=lambda p: (p[0], -1 if p[1] is None else p[1]))


def normalize_config(config):
    if not isinstance(config, dict) or set(config) != {"pushes"}:
        raise ValueError("A request configuration has exactly one field: pushes")
    return {"pushes": normalize_pushes(config["pushes"])}


class PresetStore:
    """One immutable lineage table, sharing research.sqlite3 with job records."""

    def __init__(self, state_dir, identity):
        self.identity = normalize_identity(identity)
        destination = Path(state_dir)
        self.path = destination if destination.suffix in (".sqlite", ".sqlite3", ".db") else destination / "research.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=10)
        with self._db:
            self._db.execute("CREATE TABLE IF NOT EXISTS research_presets "
                             "(id TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at REAL NOT NULL)")
            self._db.execute("CREATE TRIGGER IF NOT EXISTS research_presets_no_update "
                             "BEFORE UPDATE ON research_presets BEGIN SELECT RAISE(ABORT, 'immutable preset'); END")
            self._db.execute("CREATE TRIGGER IF NOT EXISTS research_presets_no_delete "
                             "BEFORE DELETE ON research_presets BEGIN SELECT RAISE(ABORT, 'immutable preset'); END")

    def close(self):
        with self._lock:
            self._db.close()

    def save(self, pushes, *, parents=(), evaluation=None, label=""):
        if (not isinstance(parents, (tuple, list)) or len(parents) > 64 or
                any(not isinstance(parent, str) for parent in parents) or
                len(set(parents)) != len(parents)):
            raise ValueError("Provide at most 64 distinct parent preset IDs")
        if not isinstance(label, str) or len(label) > 200:
            raise ValueError("Preset label exceeds 200 characters")
        config = {"pushes": normalize_pushes(pushes)}
        if evaluation is not None and not isinstance(evaluation, dict):
            raise ValueError("Evaluation provenance must be an object")
        parent_ids = sorted(parents)
        with self._lock:
            for parent_id in parent_ids:
                if self.get(parent_id)["identity"] != self.identity:
                    raise ValueError("Parent preset has incompatible identity")
            payload = canonical({"schema": "synapse.preset/v1", "identity": self.identity,
                                 "config": config, "parents": parent_ids,
                                 "evaluation": evaluation, "label": label})
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            preset_id = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            with self._db:
                self._db.execute("INSERT OR IGNORE INTO research_presets VALUES (?, ?, ?)",
                                 (preset_id, encoded, time.time()))
            return self.get(preset_id)

    def get(self, preset_id):
        if not isinstance(preset_id, str) or len(preset_id) != 64 or any(c not in "0123456789abcdef" for c in preset_id):
            raise ValueError("Invalid preset ID")
        with self._lock:
            row = self._db.execute("SELECT payload, created_at FROM research_presets WHERE id=?", (preset_id,)).fetchone()
        if row is None:
            raise KeyError(preset_id)
        payload = json.loads(row[0])
        if hashlib.sha256(row[0].encode("utf-8")).hexdigest() != preset_id:
            raise ValueError("Preset payload hash does not match its immutable identity")
        return {"id": preset_id, **payload, "created_at": row[1]}

    def list(self, *, limit=100):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Preset list limit must be 1..1000")
        with self._lock:
            ids = [row[0] for row in self._db.execute(
                "SELECT id FROM research_presets ORDER BY created_at, id LIMIT ?", (limit,))]
            return [self.get(preset_id) for preset_id in ids]

    def ancestry(self, preset_id):
        visited, result, pending = set(), [], [(preset_id, False)]
        while pending:
            current, expanded = pending.pop()
            if expanded:
                result.append(self.get(current))
            elif current not in visited:
                if len(visited) >= 4096:
                    raise ValueError("Preset ancestry exceeds 4096 records")
                visited.add(current)
                record = self.get(current)
                pending.append((current, True))
                pending.extend((parent, False) for parent in reversed(record["parents"]))
        return result

    def compare(self, left, right):
        a, b = self.get(left), self.get(right)
        return {"left": a, "right": b, "compatible": a["identity"] == b["identity"],
                "same_config": a["config"] == b["config"]}

    def compose(self, preset_ids, *, evaluation=None, label=""):
        if not isinstance(preset_ids, (list, tuple)) or not 1 <= len(preset_ids) <= 64:
            raise ValueError("Compose needs 1..64 preset IDs")
        merged = []
        for preset_id in preset_ids:
            config = self.replay(preset_id)
            for push in config["pushes"]:
                if push not in merged:
                    merged.append(push)
        # Validation rejects different or whole-layer overlapping edits explicitly.
        return self.save(normalize_pushes(merged), parents=sorted(set(preset_ids)),
                         evaluation=evaluation, label=label)

    def replay(self, preset_id, *, identity=None):
        expected = self.identity if identity is None else normalize_identity(identity)
        record = self.get(preset_id)
        if record["identity"] != expected:
            raise ValueError("Preset checkpoint/runtime/calibration identity is incompatible")
        return canonical(record["config"])

    def restore(self, preset_id=None, *, identity=None):
        """Return baseline or an ancestor config for the next clean runtime request."""
        if preset_id is None:
            if identity is not None and normalize_identity(identity) != self.identity:
                raise ValueError("Restore identity is incompatible")
            return {"pushes": []}
        return self.replay(preset_id, identity=identity)
