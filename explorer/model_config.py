"""Checkpoint identity and storage. A process owns exactly one checkpoint."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = os.getenv("EXPLORER_MODEL", "Qwen/Qwen3.5-4B")
REVISION = os.getenv("EXPLORER_REVISION", "main")
NO_MODEL = os.getenv("EXPLORER_NO_MODEL", "0").lower() in {"1", "true", "yes"}
BUNDLED = os.getenv("EXPLORER_BUNDLED_CACHE", "0") == "1"
GGUF = MODEL.lower().endswith(".gguf")
if BUNDLED and (MODEL != "Qwen/Qwen3.5-4B" or not NO_MODEL):
    raise ValueError("Bundled author data is a Qwen3.5-4B cache-only exhibit; never live calibration.")


def describe_config(c):
    c = c.get_text_config() if hasattr(c, "get_text_config") else c
    def get(*names, default=None):
        return next((getattr(c, n) for n in names if getattr(c, n, None) is not None), default)
    if get("is_encoder_decoder", default=False):
        raise ValueError("Encoder-decoder models need an encoder/decoder coordinate adapter.")
    if get("num_local_experts", "num_experts", "n_routed_experts", default=1) > 1:
        raise ValueError("MoE models need expert-specific coordinates; dense neuron indices would be misleading.")
    layers, hidden = get("num_hidden_layers", "n_layer"), get("hidden_size", "n_embd", "d_model")
    inter = get("intermediate_size", "ffn_dim", "n_inner", default=4 * hidden if hidden else None)
    if not all(isinstance(x, int) and x > 0 for x in (layers, hidden, inter)):
        raise ValueError("Cannot identify a dense decoder's layer, hidden and MLP dimensions.")
    heads = get("num_attention_heads", "n_head", default=0)
    return {"n_layers": layers, "hidden": hidden, "inter": inter,
            "layer_types": list(get("layer_types", default=None) or ["full_attention"] * layers),
            "attn_heads": heads, "kv_heads": get("num_key_value_heads", default=heads),
            "linear_k_heads": get("linear_num_key_heads", default=0),
            "linear_v_heads": get("linear_num_value_heads", default=0),
            "model_type": get("model_type", default="unknown")}


def checkpoint_identity(model, revision, config=None):
    """Remote commits or local file metadata isolate edits, revisions, names and calibration."""
    p = Path(model)
    files = []
    if p.is_dir():
        files = [(str(f.relative_to(p)), f.stat().st_size, f.stat().st_mtime_ns)
                 for f in sorted(p.rglob("*")) if f.is_file() and
                 (f.suffix in {".safetensors", ".bin", ".gguf"} or f.name in {"config.json", "tokenizer.json"})]
        model = str(p.resolve())
    resolved = getattr(config, "_commit_hash", None) or revision
    identity = {"model": model, "revision": resolved, "local_files": files, "schema": 1,
                "precision": os.getenv("EXPLORER_DTYPE", "cpu-float32/accelerator-auto")}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return identity, key


def configure():
    if BUNDLED:
        arch = {"n_layers": 32, "hidden": 2560, "inter": 9216, "attn_heads": 16, "kv_heads": 4,
                "linear_k_heads": 16, "linear_v_heads": 32, "model_type": "qwen3_5_text",
                "layer_types": ["linear_attention"] * 3 + ["full_attention"]}
        arch["layer_types"] *= 8
        identity, key = {"model": MODEL, "revision": "author-bundle-f79b2196", "schema": 1}, "author-qwen35-4b"
        cache = ROOT / "explorer/cache"
        config = None
    elif GGUF:
        from gguf_metadata import describe, identity as gguf_identity
        path = Path(MODEL).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"GGUF file does not exist: {path}")
        config, arch = None, describe(path)
        identity, key = gguf_identity(path)
        cache = ROOT / ".synapse/models" / key / "cache"
    else:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
        arch = describe_config(config)
        identity, key = checkpoint_identity(MODEL, REVISION, config)
        cache = ROOT / ".synapse/models" / key / "cache"
    state = ROOT / ".synapse/models" / key
    state.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    (state / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    return config, arch, identity, key, state, cache


CONFIG, ARCH, IDENTITY, MODEL_KEY, STATE, CACHE = configure()
