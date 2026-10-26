"""Read GGUF metadata without loading/dequantizing weights or importing a runtime."""

import hashlib
import json
import struct
from pathlib import Path


def metadata(path):
    with open(path, "rb") as f:
        size = Path(path).stat().st_size

        def read(n):
            if n < 0 or n > size - f.tell():
                raise ValueError("Truncated or malformed GGUF metadata")
            b = f.read(n)
            if len(b) != n:
                raise ValueError("Truncated GGUF")
            return b

        def number(fmt):
            return struct.unpack("<" + fmt, read(struct.calcsize(fmt)))[0]

        def value(kind, keep=True):
            if kind == 8:
                n = number("Q")
                if not keep:
                    if n > size - f.tell():
                        raise ValueError("Truncated GGUF string")
                    f.seek(n, 1)
                    return None
                if n > 1 << 20:
                    raise ValueError("GGUF metadata string too large")
                return read(n).decode("utf-8")
            if kind == 9:
                subtype, n = number("I"), number("Q")
                if subtype == 9 or n > 10_000_000:
                    raise ValueError("Unsupported GGUF metadata array")
                if keep and n < 10000:
                    return [value(subtype) for _ in range(n)]
                for _ in range(n):
                    value(subtype, False)
                return None
            formats = {
                0: "B",
                1: "b",
                2: "H",
                3: "h",
                4: "I",
                5: "i",
                6: "f",
                7: "?",
                10: "Q",
                11: "q",
                12: "d",
            }
            if kind not in formats:
                raise ValueError("Unknown GGUF metadata type")
            v = number(formats[kind])
            return v if keep else None

        if read(4) != b"GGUF" or number("I") != 3:
            raise ValueError("Synapse expects a GGUF v3 file")
        tensors, count = number("Q"), number("Q")
        if count > 100000 or tensors > 1000000:
            raise ValueError("Unreasonable GGUF header")
        result = {}
        for _ in range(count):
            key = value(8)
            v = value(
                number("I"),
                not key.startswith("tokenizer.") or key == "tokenizer.chat_template",
            )
            if v is not None:
                result[key] = v
        return result


def describe(path):
    m = metadata(path)
    family = m.get("general.architecture")
    if family != "qwen35":
        raise ValueError(
            f"GGUF architecture {family!r} has no validated Synapse neuron adapter yet; currently qwen35 is supported."
        )
    get = lambda key: m[f"{family}.{key}"]
    layers = get("block_count")
    inter = get("feed_forward_length")
    if not isinstance(inter, int) or m.get(f"{family}.expert_count", 0):
        raise ValueError("GGUF expert/nonuniform MLPs need another coordinate adapter")
    interval = get("full_attention_interval")
    return {
        "n_layers": layers,
        "hidden": get("embedding_length"),
        "inter": inter,
        "model_type": family,
        "attn_heads": get("attention.head_count"),
        "kv_heads": get("attention.head_count_kv"),
        "linear_k_heads": m.get(f"{family}.ssm.group_count", 0),
        "linear_v_heads": m.get(f"{family}.ssm.time_step_rank", 0),
        "layer_types": [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(layers)
        ],
    }


def identity(path):
    p = Path(path).expanduser().resolve()
    s = p.stat()
    data = {
        "model": str(p),
        "revision": "local-gguf",
        "local_files": [(p.name, s.st_size, s.st_mtime_ns)],
        "schema": 2,
        "precision": "gguf-native",
        "backend": "llama.cpp-b11669/synapse-callback-v1",
    }
    return data, hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[
        :20
    ]
