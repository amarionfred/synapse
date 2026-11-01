"""Strict bounded paired-experiment inputs; no model loading or data discovery.

One job contains one split. Generation receives prompts without evaluator
labels. Sequence scoring measures complete-answer likelihood, not accuracy.
"""
from copy import deepcopy
import hashlib
import json
import math

SCHEMA = "synapse.experiment.v1"
SPLITS = {"development", "validation", "final"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _object(value, keys, label):
    if type(value) is not dict or set(value) - set(keys):
        raise ValueError(f"{label} must be an object with only {sorted(keys)}")
    return value


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in {low}..{high}")
    return value


def _number(value, low, high, label):
    if not finite_number(value) or not low < value <= high:
        raise ValueError(f"{label} must be finite and in ({low}, {high}]")
    return float(value)


def finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _text(value, low, high, label):
    if not isinstance(value, str) or not low <= len(value) <= high:
        raise ValueError(f"{label} must contain {low}..{high} characters")
    return value


def configuration(value):
    value = _object(value, {"pushes"}, "configuration")
    pushes = value.get("pushes", [])
    if type(pushes) is not list or len(pushes) > 32:
        raise ValueError("pushes must be a list of at most 32 interventions")
    out = []
    for push in pushes:
        if type(push) not in (tuple, list) or len(push) != 4:
            raise ValueError("Each push must be [layer, neuron, dose, mode]")
        layer, neuron, dose, mode = push
        _integer(layer, 0, 4095, "push layer")
        if neuron is not None:
            _integer(neuron, 0, 1_000_000, "push neuron")
        if not finite_number(dose):
            raise ValueError("push dose must be finite")
        if type(mode) is not str or mode not in {"add", "mul"} or (neuron is None and mode != "mul"):
            raise ValueError("push mode must be add or mul; whole layers multiply")
        out.append([layer, neuron, float(dose), mode])
    return {"pushes": out}


def experiment_spec(value):
    """Validate and freeze a JSON-native spec without loading a model."""
    value = deepcopy(_object(value, {
        "schema", "name", "mode", "split", "seed", "items", "baseline",
        "treatment", "limits", "max_new", "chat", "prefix", "eos_policy",
        "candidate_frozen",
    }, "experiment"))
    if value.get("schema", SCHEMA) != SCHEMA:
        raise ValueError("Unsupported experiment schema")
    mode = value.get("mode", "generation")
    if type(mode) is not str or mode not in {"generation", "sequence_score"}:
        raise ValueError("mode must be generation or sequence_score")
    split = value.get("split")
    if type(split) is not str or split not in SPLITS:
        raise ValueError("Choose exactly one split: development, validation or final")
    frozen = value.get("candidate_frozen", False)
    if type(frozen) is not bool or (split == "final" and not frozen):
        raise ValueError("Final evaluation requires candidate_frozen=true")
    items = value.get("items")
    if type(items) is not list or not 1 <= len(items) <= 64:
        raise ValueError("items must contain 1..64 evaluator items")
    clean, seen = [], set()
    for item in items:
        _object(item, {"id", "prompt", "answer", "family", "seed"}, "item")
        ident = _text(item.get("id"), 1, 128, "item id")
        if ident in seen:
            raise ValueError("Item ids must be unique")
        seen.add(ident)
        row = {"id": ident,
               "prompt": _text(item.get("prompt"), 1, 32768, "item prompt"),
               "answer": _text(item.get("answer"), 1, 4096, "item answer")}
        if "family" in item:
            row["family"] = _text(item["family"], 1, 128, "item family")
        if "seed" in item:
            row["seed"] = _integer(item["seed"], 0, 2**63 - 1, "item seed")
        clean.append(row)
    limits = _object(value.get("limits"), {
        "max_requests", "max_tokens", "max_wall_seconds",
    }, "limits")
    clean_limits = {
        "max_requests": _integer(limits.get("max_requests"), 1, 512, "max_requests"),
        "max_tokens": _integer(limits.get("max_tokens"), 1, 1_048_576, "max_tokens"),
        "max_wall_seconds": _number(limits.get("max_wall_seconds"), 0, 3600,
                                    "max_wall_seconds"),
    }
    chat = value.get("chat", True)
    if type(chat) is not bool:
        raise ValueError("chat must be boolean")
    if mode == "generation" and not chat:
        raise ValueError("generation uses a user chat turn; chat=false is score-only")
    eos = value.get("eos_policy", "exclude")
    if type(eos) is not str or eos not in {"exclude", "include"}:
        raise ValueError("eos_policy must be exclude or include")
    prefix = _text(value.get("prefix", ""), 0, 4096, "prefix")
    if mode == "generation" and prefix:
        raise ValueError("prefix is score-only")
    out = {
        "schema": SCHEMA, "name": _text(value.get("name", "Paired experiment"), 1, 200, "name"),
        "mode": mode, "split": split,
        "seed": _integer(value.get("seed", 0), 0, 2**63 - 1, "seed"),
        "items": clean, "baseline": configuration(value.get("baseline", {})),
        "treatment": configuration(value.get("treatment", {})),
        "limits": clean_limits,
        "max_new": _integer(value.get("max_new", 32), 1, 256, "max_new"),
        "chat": chat, "prefix": prefix, "eos_policy": eos,
        "candidate_frozen": frozen,
    }
    if len(canonical(out).encode()) > 256 * 1024:
        raise ValueError("Experiment exceeds the 256 KiB input limit")
    return out


def grade_exact(text, answer):
    """Exact whole-answer equality, ignoring surrounding whitespace only."""
    if not isinstance(text, str) or not isinstance(answer, str):
        raise ValueError("Exact grading requires text and reference strings")
    return text.strip() == answer.strip()
