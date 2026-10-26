"""Explicit local Qwen integration check. Run with the explorer/chat services stopped.

Loads native contexts sequentially, keeps outputs in evidence, and never writes weights.
"""

import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explorer"))
from llama_bridge import Bridge, Tokenizer

path = ROOT / "models/Qwen3.5-4B-Q4_K_M.gguf"
out = ROOT / "evidence/gguf-native-verification.json"
result = {
    "purpose": "native callback equivalence and intervention plumbing; no capability claim"
}
start = time.monotonic()


def save():
    out.write_text(json.dumps(result, indent=2) + "\n")


try:
    bare = Bridge(path, hooks=False)
    ids = bare.tokenize("The capital of France is")
    baseline = bare.run(ids)[0].copy()
    bare.close()
    hooked = Bridge(path)
    tok = Tokenizer(hooked)
    observed, act, res = hooked.run(ids, read=(14, 8949), residual=True, observe=True)
    zero, _, _ = hooked.run(ids, pushes=[(14, 8949, "add", 0.0)])
    changed, _, _ = hooked.run(ids, pushes=[(14, 8949, "add", 5.0)])
    again = hooked.run(ids)[0]
    result.update(
        tokens=ids,
        activation=act.tolist(),
        residual_shape=list(res.shape),
        finite=bool(
            np.isfinite(observed).all()
            and np.isfinite(act).all()
            and np.isfinite(res).all()
        ),
        observed_max_logit_difference=float(np.max(np.abs(baseline - observed))),
        zero_max_logit_difference=float(np.max(np.abs(baseline - zero))),
        reset_max_logit_difference=float(np.max(np.abs(baseline - again))),
        nonzero_max_logit_difference=float(np.max(np.abs(baseline - changed))),
        baseline_top=tok.decode([int(baseline.argmax())]),
        column_norm=float(np.linalg.norm(hooked.column(14, 8949))),
    )
    save()
    assert result["finite"]
    np.testing.assert_allclose(observed, baseline, atol=0.0005, rtol=0.0001)
    np.testing.assert_allclose(zero, baseline, atol=0.0005, rtol=0.0001)
    np.testing.assert_allclose(again, baseline, atol=0.0005, rtol=0.0001)
    assert result["nonzero_max_logit_difference"] > 0.001
    # Multi-ubatch capture must collect every token in the right order, all 32 layers.
    long_ids = hooked.tokenize("The river flows past the quiet town. " * 9)
    _, full, _ = hooked.run(long_ids, full=True)
    _, single, _ = hooked.run(long_ids, read=(14, 8949))
    np.testing.assert_allclose(full[14, :, 8949], single, atol=1e-6, rtol=1e-5)
    result["full_capture_shape"] = list(full.shape)
    result["multi_ubatch_capture_equal"] = True
    lens = hooked.lens(14, 8949)
    assert np.isfinite(lens).all()
    result["lens_top"] = [tok.decode([int(i)]) for i in lens.argsort()[-8:][::-1]]
    result["chat_template"] = tok.decode(
        tok.apply_chat_template([{"role": "user", "content": "Hi"}])
    )
    hooked.close()
    result["passed"] = True
except Exception as e:
    result["error"] = repr(e)
    raise
finally:
    result["elapsed_seconds"] = time.monotonic() - start
    save()
    print(json.dumps(result, indent=2), flush=True)
