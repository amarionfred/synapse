"""Small, checkpoint-bound research artifacts; no inference or saved-weight edits."""
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile
import numpy as np


def bounded_strength(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Projection strength must be a number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError("Projection strength is out of range") from exc
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError("Projection strength must be within [-1, 1]")
    return value


def unit_direction(value, hidden):
    try:
        u = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Direction must be a numeric vector") from exc
    if u.shape != (hidden,) or not np.isfinite(u).all():
        raise ValueError(f"Direction must contain {hidden} finite values")
    norm = float(np.linalg.norm(u))
    if not math.isfinite(norm) or abs(norm - 1) > 5e-5:
        raise ValueError("Direction must already have unit norm")
    return np.ascontiguousarray(u, dtype=np.float32)


def contrastive_direction(refused, answered):
    """Only call with baseline-confirmed development examples, never held-out data."""
    r, a = np.asarray(refused, dtype=np.float64), np.asarray(answered, dtype=np.float64)
    if (r.ndim != 2 or a.shape != r.shape or r.shape[0] < 3 or r.shape[1] < 1
            or not np.isfinite(r).all() or not np.isfinite(a).all()):
        raise ValueError("Need at least three matched, finite residual pairs")
    d = r.mean(axis=0) - a.mean(axis=0)
    norm = float(np.linalg.norm(d))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("No finite nonzero contrastive direction")
    return unit_direction(d / norm, r.shape[1])


def projection_request(body, model_key, hidden, layers):
    if body.get("model_key") != model_key:
        raise ValueError("Projection checkpoint key does not match the loaded model")
    text = body.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > 12000:
        raise ValueError("Use nonempty text up to 12000 characters")
    layer = body.get("layer", 21)
    if type(layer) is not int or not 0 <= layer < layers:
        raise ValueError("Invalid residual capture layer")
    max_new = body.get("max_new", 32)
    if type(max_new) is not int or not 0 <= max_new <= 128:
        raise ValueError("Residual research allows 0..128 generated tokens")
    projection = body.get("projection")
    u, strength = None, 0.0
    if projection is not None:
        if not isinstance(projection, dict):
            raise ValueError("Projection must contain a direction and strength")
        strength = bounded_strength(projection.get("strength", 0.0))
        if "artifact" in projection:
            if "direction" in projection or not isinstance(projection["artifact"], str):
                raise ValueError("Use either an artifact path or a direction")
        else:
            u = unit_direction(projection.get("direction"), hidden)
    return text, layer, max_new, u, strength


def writer_targets(arch):
    """Linear-algebra W shapes are [hidden,input]; GGUF stores each W row."""
    for layer, kind in enumerate(arch["layer_types"]):
        if kind not in {"full_attention", "linear_attention"}:
            raise ValueError("Unsupported residual writer type")
        attention = "attn_output" if kind == "full_attention" else "ssm_out"
        yield f"blk.{layer}.{attention}.weight"
        yield f"blk.{layer}.ffn_down.weight"


def export_factors(bridge, direction, destination, provenance):
    """Synapse research NPZ, not a llama.cpp LoRA. Embedding stays input-only.

    Factors represent corrections for the 64 block writers only. The shared B
    avoids redundant copies. Native inference must still validate numerical and
    behavioral agreement; exporting is not applying an adapter.
    """
    u = unit_direction(direction, bridge.arch["hidden"])
    destination = Path(destination)
    if destination.suffix != ".npz" or destination.exists():
        raise ValueError("Choose a new .npz artifact path")
    if provenance.get("model_key") != bridge.model_key or not provenance.get("direction_source"):
        raise ValueError("Factor export requires checkpoint and direction provenance")
    # Explicit widths are taken from this architecture, never from other models.
    if bridge.arch["model_type"] != "qwen35":
        raise ValueError("Only the qwen35 writer map is implemented")
    arrays = {"B": -u}
    specs = []
    # Attention widths vary by model size: inspect actual GGUF dimensions below.
    for name in writer_targets(bridge.arch):
        shape = bridge.weight_shape(name)
        if len(shape) != 4 or shape[1:] != (len(u), 1, 1):
            raise ValueError(f"Unsupported residual writer shape: {name}")
        width = shape[0]
        a = np.zeros(width, np.float64)
        for start in range(0, len(u), 64):
            count = min(64, len(u) - start)
            rows = bridge.weight_rows(name, start, count, width=width)
            a += u[start:start + count].astype(np.float64) @ rows.astype(np.float64)
        arrays[name] = a.astype(np.float32)
        specs.append({"name": name, "shape": [len(u), width], "A_shape": [width]})
    manifest = {
        "format": "synapse.rank-one-research.v1", "provenance": provenance,
        "operation": "W + strength * outer(B, A)", "targets": specs,
        "B_shape": [len(u)],
        "input_embedding": "requires input-only runtime projection",
        "readout": "unchanged", "native_adapter": "synapse-callback-rank-one-v1",
        "llama_cpp_lora": False,
        "identity": bridge.identity,
        "direction_sha256": hashlib.sha256(u.tobytes()).hexdigest(),
    }
    arrays["manifest"] = np.array(json.dumps(manifest, sort_keys=True))
    # Exclusive creation prevents replacing a previous experiment artifact.
    with destination.open("xb") as f:
        np.savez(f, **arrays)
    return manifest


def artifact_path(value):
    root = Path(__file__).resolve().parents[1]
    base = (root / ".synapse/experiments").resolve()
    if not isinstance(value, str) or not value:
        raise ValueError("Artifact path must be a nonempty string")
    path = (root / value).resolve()
    if not path.is_relative_to(base) or path.suffix != ".npz":
        raise ValueError("Artifacts must be .npz files under .synapse/experiments")
    return path


def load_factors(bridge, value):
    """Validate bounded NPZ members before NumPy can allocate from their headers."""
    path = artifact_path(value)
    limit = 16 * 1024 * 1024
    try:
        with path.open("rb") as f:
            raw = f.read(limit + 1)
    except OSError as exc:
        raise ValueError(f"Cannot read factor archive: {exc}") from exc
    if len(raw) > limit:
        raise ValueError("Factor archive exceeds 16 MiB")
    arrays = {}
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            members = zf.infolist()
            if len(members) > 130 or sum(m.file_size for m in members) > limit:
                raise ValueError("Factor archive expands beyond the bounded budget")
            for member in members:
                if not member.filename.endswith(".npy") or member.filename in arrays:
                    raise ValueError("Unexpected or duplicate factor archive member")
                with zf.open(member) as f:
                    payload = f.read(limit + 1)
                if len(payload) > limit:
                    raise ValueError("Oversized factor archive member")
                stream = io.BytesIO(payload)
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("Unsupported factor array format")
                if (dtype.hasobject or len(shape) > 2 or fortran
                        or math.prod(shape) * dtype.itemsize != len(payload) - stream.tell()):
                    raise ValueError("Unsafe or malformed factor array header")
                arrays[member.filename] = np.frombuffer(payload, dtype=dtype,
                                                        offset=stream.tell()).reshape(shape)
        manifest_array = arrays["manifest.npy"]
        if manifest_array.shape != () or manifest_array.dtype.kind != "U" or manifest_array.nbytes > 65536:
            raise ValueError("Invalid factor manifest")
        manifest = json.loads(str(manifest_array))
        if (manifest["format"] != "synapse.rank-one-research.v1"
                or manifest["provenance"]["model_key"] != bridge.model_key
                or manifest["identity"] != json.loads(json.dumps(bridge.identity))
                or manifest["operation"] != "W + strength * outer(B, A)"
                or manifest["readout"] != "unchanged"
                or manifest["native_adapter"] != "synapse-callback-rank-one-v1"
                or manifest["llama_cpp_lora"] is not False
                or manifest["input_embedding"] != "requires input-only runtime projection"):
            raise ValueError("Factor manifest does not match this checkpoint and operation")
        b = arrays["B.npy"]
        if b.dtype != np.dtype("float32"):
            raise ValueError("Factors must use float32")
        u = unit_direction(-b, bridge.arch["hidden"])
        if hashlib.sha256(u.tobytes()).hexdigest() != manifest["direction_sha256"]:
            raise ValueError("Factor direction checksum differs")
        expected = list(writer_targets(bridge.arch))
        if ([entry["name"] for entry in manifest["targets"]] != expected
                or manifest["B_shape"] != [len(u)]):
            raise ValueError("Factor manifest target order differs")
        if set(arrays) != {"B.npy", "manifest.npy"} | {name + ".npy" for name in expected}:
            raise ValueError("Factor target set differs from the checkpoint")
        factors = []
        for name, spec in zip(expected, manifest["targets"]):
            width, hidden, d2, d3 = bridge.weight_shape(name)
            a = arrays[name + ".npy"]
            if (a.dtype != np.dtype("float32") or a.shape != (width,)
                    or (hidden, d2, d3) != (len(u), 1, 1) or not np.isfinite(a).all()):
                raise ValueError("Factor shape or values differ from the checkpoint")
            if spec["shape"] != [hidden, width] or spec["A_shape"] != [width]:
                raise ValueError("Factor manifest shape differs")
            factors.append(a)
        flat = np.ascontiguousarray(np.concatenate(factors), dtype=np.float32)
        if flat.size > 2_000_000:
            raise ValueError("Factor bank exceeds native bound")
        return u, flat, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                         "manifest": manifest}
    except (KeyError, TypeError, zipfile.BadZipFile, EOFError, UnicodeError) as exc:
        raise ValueError("Malformed factor archive") from exc
