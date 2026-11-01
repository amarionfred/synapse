"""Bounded MLP scopes: masks select processed prompt tokens/decode input steps.

Capture is before edits at the MLP output-projection input. An omitted phase mask
selects that phase; [] selects none. decode_steps[0] is the first generated token
fed back to predict the second output token. No attention layout is assumed.
"""
import hashlib
import json
import math

MAX_SELECTED = 4096


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def bounded_int(value, name, low=0, high=2047):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return value


def selection_rows(runtime, selection, token_count):
    if not isinstance(selection, (list, tuple)) or not 1 <= len(selection) <= MAX_SELECTED:
        raise ValueError("Capture requires 1..4096 selected MLP scalar coordinates")
    rows = []
    for item in selection:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError("Capture coordinates are [layer, position, neuron]")
        layer, position, neuron = item
        bounded_int(layer, "layer")
        bounded_int(neuron, "neuron", high=1_000_000)
        bounded_int(position, "position", high=token_count - 1)
        runtime.validate_coordinate(layer, neuron)
        rows.append((layer, position, neuron))
    if len(set(rows)) != len(rows):
        raise ValueError("Duplicate capture coordinate")
    return rows


class Plan:
    def __init__(self, runtime, prompt_ids, pushes=None, interventions=None, patches=None):
        self.prompt_count = len(prompt_ids)
        raw = list(pushes or [])
        if len(raw) > 128:
            raise ValueError("At most 128 global interventions")
        self.global_pushes = runtime._research_normalize(raw)
        self.scoped = []
        occupied = {(p[0], p[1]) for p in self.global_pushes}
        if interventions is not None and not isinstance(interventions, list):
            raise ValueError("interventions must be a list")
        if len(interventions or []) > 128:
            raise ValueError("At most 128 scoped interventions")
        for item in interventions or []:
            if not isinstance(item, dict) or set(item) - {
                    "layer", "n", "dose", "mode", "prompt_positions", "decode_steps"}:
                raise ValueError("Unsupported scoped MLP intervention")
            layer, neuron, dose, mode = (item.get(k) for k in ("layer", "n", "dose", "mode"))
            bounded_int(layer, "layer")
            bounded_int(neuron, "n", high=1_000_000)
            if (layer, neuron) in occupied or (layer, None) in occupied:
                raise ValueError("Conflicting global/scoped intervention coordinates")
            occupied.add((layer, neuron))
            normalized = runtime._research_normalize([(layer, neuron, dose, mode)])[0]
            masks = []
            for key, limit in (("prompt_positions", self.prompt_count - 1), ("decode_steps", 1023)):
                values = item.get(key)
                if values is not None:
                    if not isinstance(values, list) or len(values) > MAX_SELECTED:
                        raise ValueError(f"{key} must be a bounded list")
                    values = {bounded_int(v, key, high=limit) for v in values}
                masks.append(values)
            self.scoped.append((normalized, *masks))
        self.patch_rows = []
        if patches is not None:
            if not isinstance(patches, dict) or set(patches) != {"reference", "target_token_ids"}:
                raise ValueError("Patches require a matched reference and explicit target_token_ids")
            reference = patches["reference"]
            if not isinstance(reference, dict) or reference.get("kind") != "mlp-selected-v1" or not reference.get("completed"):
                raise ValueError("Unsupported or incomplete patch reference")
            payload = {k: v for k, v in reference.items() if k != "sha256"}
            if reference.get("sha256") != digest(payload):
                raise ValueError("Patch reference hash mismatch")
            if reference.get("identity") != runtime.research_identity():
                raise ValueError("Patch reference checkpoint/runtime/calibration mismatch")
            if patches["target_token_ids"] != list(prompt_ids) or len(reference.get("prompt_token_ids", [])) != len(prompt_ids):
                raise ValueError("Patch reference and declared target must be matched token sequences")
            rows = selection_rows(runtime, reference.get("selection"), len(prompt_ids))
            values = reference.get("values")
            if not isinstance(values, list) or len(values) != len(rows):
                raise ValueError("Patch reference values do not match selection")
            for (layer, position, neuron), value in zip(rows, values):
                if (layer, neuron) in occupied or (layer, None) in occupied:
                    raise ValueError("Patch and steering coordinates conflict")
                if type(value) not in (float, int) or not math.isfinite(value) or abs(value) > 1e30:
                    raise ValueError("Patch values must be finite bounded scalars")
                self.patch_rows.append((layer, neuron, "set", float(value), position))

    def edits(self, start, count):
        out = [r for r in self.patch_rows if start <= r[4] < start + count]
        for (layer, neuron, mode, value), prompt, decode in self.scoped:
            for position in range(start, start + count):
                selected = ((prompt is None or position in prompt) if position < self.prompt_count
                            else (decode is None or position - self.prompt_count in decode))
                if selected:
                    out.append((layer, neuron, mode, value, position))
        if len(out) > MAX_SELECTED:
            raise ValueError("Selected edits exceed the 4096-scalar per-forward budget")
        return out
