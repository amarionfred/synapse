"""Real-checkpoint connection/reference checks; run with model services stopped."""

import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/Qwen3.5-4B-Q4_K_M.gguf"
os.environ.update(
    EXPLORER_MODEL=str(MODEL), EXPLORER_BUNDLED_CACHE="0", EXPLORER_NO_MODEL="0"
)
sys.path.insert(0, str(ROOT / "explorer"))
from llama_bridge import Bridge, Tokenizer
from connection_graph import rank_connections
from gguf_runtime import GGUFRuntime
import torch

torch.set_num_threads(4)

result = {
    "purpose": "quantized-weight arithmetic and integration verification, not a circuit/capability claim"
}
receipt = ROOT / "evidence/gguf-connections-native.json"


def record():
    receipt.write_text(json.dumps(result, indent=2) + "\n")


started = time.monotonic()
b = None
try:
    b = Bridge(MODEL)
    ids = b.tokenize("The capital of France is")
    before = b.run(ids)[0].copy()
    tick = time.monotonic()
    g, u = b.connection_vectors(14, 8949)
    result["connection_seconds"] = time.monotonic() - tick
    result["shape"] = list(g.shape)
    assert np.isfinite(g).all() and np.isfinite(u).all()
    print("Native connection vectors ready", result["connection_seconds"], flush=True)
    record()
    # Independent NumPy float64 matrix-vector products, exported in bounded blocks.
    gain = b.weight_rows("blk.14.post_attention_norm.weight", 0, 1)[0]
    rg = b.weight_rows("blk.14.ffn_gate.weight", 8949, 1)[0] * gain
    ru = b.weight_rows("blk.14.ffn_up.weight", 8949, 1)[0] * gain
    expected_g, expected_u = np.zeros(b.arch["inter"]), np.zeros(b.arch["inter"])
    for h in range(0, b.arch["hidden"], 64):
        w = b.weight_rows(
            "blk.13.ffn_down.weight", h, min(64, b.arch["hidden"] - h), b.arch["inter"]
        ).astype(np.float64)
        expected_g += rg[h : h + len(w)].astype(float) @ w
        expected_u += ru[h : h + len(w)].astype(float) @ w
    np.testing.assert_allclose(g[13], expected_g, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(u[13], expected_u, atol=2e-6, rtol=2e-5)
    result["upstream_reference_max_error"] = max(
        float(np.max(np.abs(g[13] - expected_g))),
        float(np.max(np.abs(u[13] - expected_u))),
    )
    v = b.column(14, 8949) * b.weight_rows("blk.15.post_attention_norm.weight", 0, 1)[0]
    for n in range(0, b.arch["inter"], 64):
        for kind, actual in (("gate", g), ("up", u)):
            w = b.weight_rows(
                f"blk.15.ffn_{kind}.weight", n, min(64, b.arch["inter"] - n)
            ).astype(np.float64)
            reference = w @ v.astype(np.float64)
            np.testing.assert_allclose(
                actual[15, n : n + len(w)], reference, atol=2e-6, rtol=2e-5
            )
    result["downstream_reference_passed"] = True
    ranked = rank_connections(g, u, 14, 8)
    result["strongest"] = ranked
    reader = int(np.abs(g[15] * u[15]).argmax())
    reverse_g, reverse_u = b.connection_vectors(15, reader)
    np.testing.assert_allclose(
        [reverse_g[14, 8949], reverse_u[14, 8949]],
        [g[15, reader], u[15, reader]],
        atol=2e-6,
        rtol=2e-5,
    )
    result["reciprocal_pair_passed"] = True
    after = b.run(ids)[0]
    result["baseline_max_logit_difference"] = float(np.max(np.abs(after - before)))
    assert np.array_equal(before, after)
    print(
        "Matrix reference, reciprocal pair and baseline preservation passed", flush=True
    )
    record()
    cache = ROOT / ".synapse/verification/connections"
    rt = GGUFRuntime(str(cache / "stats"), str(cache))
    rt.model, rt.tok, rt.dev = b, Tokenizer(b), "cpu"
    old = b.lens(14, 8949)
    tick = time.monotonic()
    labels = rt.lens_many([(14, 8949), (13, 7956)], 12)
    result["batched_lens_seconds"] = time.monotonic() - tick
    assert labels[(14, 8949)]["up"] == [
        rt.tok.decode([i]) for i in np.argsort(old)[-12:][::-1]
    ]
    assert labels[(14, 8949)]["down"] == [
        rt.tok.decode([i]) for i in np.argsort(old)[:12]
    ]
    result["batched_lens_matches_scalar"] = True
    result["lens_example"] = labels[(14, 8949)]
    result["passed"] = True
finally:
    result["elapsed_seconds"] = time.monotonic() - started
    record()
    if b:
        b.close()
    print(json.dumps(result, indent=2), flush=True)
