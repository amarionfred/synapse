"""Explicit, bounded real-checkpoint test. No reserved acceptance coordinates.

Default uses a new cache and forbids inference. --reference checks the saved
inspection vectors/lens against inference in a separate process; stop model
services first and run that mode under the existing memory bound.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/Qwen3.5-4B-Q4_K_M.gguf"
os.environ.update(EXPLORER_MODEL=str(MODEL), EXPLORER_BUNDLED_CACHE="0", EXPLORER_NO_MODEL="0")
sys.path.insert(0, str(ROOT / "explorer"))

import numpy as np
from gguf_runtime import GGUFRuntime
from llama_bridge import Bridge
from model_config import IDENTITY


def available_mib():
    return int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines()
                    if x.startswith("MemAvailable:"))) / 1024


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--reference", type=Path)
    args = ap.parse_args()
    base = ROOT / ".synapse/verification"
    base.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="inspection-", dir=base))
    receipt = run / "result.json"
    result = {"checkpoint": IDENTITY, "run": str(run), "available_mib": available_mib(),
              "purpose": "weight inspection plumbing, not a causal circuit or capability claim"}
    started = time.monotonic()
    b = None
    rt = GGUFRuntime(str(run / "stats"), str(run / "cache"))

    def record():
        result.update(elapsed_seconds=time.monotonic() - started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        receipt.write_text(json.dumps(result, indent=2) + "\n")

    try:
        if args.reference:
            saved = np.load(args.reference / "reference.npz")
            # Keep the production admission and shared inference lease intact.
            rt.ensure_loaded()
            b = rt.model
            ids = b.tokenize("The capital of France is")
            before = b.run(ids)[0].copy()
            g, u = b.connection_vectors(15, 417)
            np.testing.assert_array_equal(g, saved["gate"])
            np.testing.assert_array_equal(u, saved["up"])
            np.testing.assert_array_equal(b.lens(15, 417), saved["lens"])
            after = b.run(ids)[0]
            np.testing.assert_array_equal(before, after)
            result.update(reference=str(args.reference), vector_and_lens_exact=True,
                          baseline_max_logit_difference=float(np.abs(before - after).max()))
        else:
            def forbidden():
                raise AssertionError("Inspection attempted inference loading")
            rt.ensure_loaded = forbidden
            result["inference_deliberately_unavailable"] = True
            # The explicitly reported failures plus independently chosen test points.
            coords = [(0, 113), (15, 417)] + [(31, n) for n in (7066, 7621, 4089, 2565, 6399, 7166)]
            result["selections"] = []
            for l, n in coords:
                tick = time.monotonic()
                assert rt.cached_connections(l, n).get("pending")
                assert rt.cached_lens(l, n).get("pending")
                con = rt.connections(l, n, 24, 5)
                lens = rt.lens(l, n, 12)
                graph = rt.connection_graph(l, n, 5, 2, 1)
                summary = rt.connection_summary(l, n, 5, 2, 1, 24)
                assert not con["cached"] and lens["up"] and lens["down"]
                assert len(con["up"]) == (0 if l == 0 else 24)
                assert len(con["down"]) == (0 if l == 31 else 24)
                assert rt.model is None and rt.tok is None and rt.status == "idle"
                b = rt.inspector
                assert b.lease is None and not b.lib.syn_has_context(b.handle)
                result["selections"].append({"layer": l, "n": n, "seconds": time.monotonic() - tick,
                                             "connections": con, "lens": lens, "graph": graph,
                                             "summary": summary, "inference_loaded": False})
                print(f"Cold L{l} N{n}: links/labels/web passed", flush=True)
                record()
            g, u = b.connection_vectors(15, 417)
            scalar = b.lens(15, 417)
            labels = rt.lens(15, 417, 12)
            from llama_bridge import Tokenizer
            tok = Tokenizer(b)
            assert labels["up"] == [tok.decode([i]) for i in np.argsort(scalar)[-12:][::-1]]
            assert labels["down"] == [tok.decode([i]) for i in np.argsort(scalar)[:12]]
            np.savez_compressed(run / "reference.npz", gate=g, up=u, lens=scalar)
            result["scalar_batched_lens_agree"] = True
            try:
                b.run(b.tokenize("Hello"))
            except RuntimeError as e:
                assert "no inference context" in str(e)
                result["inference_rejected_by_native_inspector"] = str(e)
            else:
                raise AssertionError("Inspector accepted inference")
        with MODEL.open("rb") as f:
            result["gguf_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
        assert result["gguf_sha256"] == "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
        result["passed"] = True
    except BaseException as e:
        result["error"] = repr(e)
        raise
    finally:
        if b:
            b.close()
        record()
        print(str(receipt), flush=True)


if __name__ == "__main__":
    main()
