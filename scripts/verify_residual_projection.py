#!/usr/bin/env python3
"""Bounded HTTP/native integration check using the already selected Synapse model.

Never loads a second context. Uses seeded random directions, NOT a refusal
direction. A successful run validates plumbing and baseline recovery only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--output", type=Path, required=True,
                        help="New experiment directory; existing results are never overwritten")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    if not args.output.resolve().is_relative_to((ROOT / ".synapse/experiments").resolve()):
        parser.error("Output must be under .synapse/experiments for the factor round-trip")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records = []
    report = {"status": "running", "purpose": "infrastructure_only",
              "refusal_direction_measured": False, "saved_weights_modified": False}

    def request(path, body=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}{path}",
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t = time.monotonic()
        entry = {"path": path, "request": body}
        try:
            with urllib.request.urlopen(req, timeout=240) as response:
                entry["response"] = json.load(response)
            return entry["response"]
        except urllib.error.HTTPError as exc:
            entry["http_error"] = exc.code
            entry["body"] = exc.read().decode(errors="replace")
            if exc.code in (404, 405) and path.endswith("/residual"):
                raise RuntimeError("The live server lacks the residual API; restart Synapse first") from exc
            raise
        finally:
            entry["elapsed_seconds"] = time.monotonic() - t
            path_out = args.output / f"request-{len(records):02d}.json"
            path_out.write_text(json.dumps(entry, indent=2) + "\n")
            records.append(path_out.name)

    try:
        overview = request("/api/agent/overview")
        key, arch = overview["model_key"], overview["arch"]
        if arch["model_type"] != "qwen35":
            raise RuntimeError("This check is scoped to the validated qwen35 graph")
        report["identity"] = overview["identity"]
        report["model_key"] = key
        hidden, layers = arch["hidden"], arch["n_layers"]
        u = np.random.default_rng(261026).normal(size=hidden)
        u = (u / np.linalg.norm(u)).astype(np.float32)
        report["direction_seed"] = 261026
        report["direction_sha256"] = hashlib.sha256(u.tobytes()).hexdigest()
        question = {"model_key": key, "text": "What is 7 + 5? Reply with the number only.",
                    "layer": min(21, layers - 1), "max_new": 8}
        baseline = request("/api/agent/residual", question)
        zero = request("/api/agent/residual", dict(question, projection={
            "direction": u.tolist(), "strength": 0.0,
        }))
        assert baseline["baseline"]["token_ids"] == zero["baseline"]["token_ids"] == zero["intervened"]["token_ids"]
        assert zero["baseline_restoration_max_abs"] <= 1e-5
        assert abs(zero["kl_baseline_to_intervened"]) <= 1e-6
        assert zero["baseline"]["text"].strip() == "12"
        # More than one 64-token native chunk and a final-layer capture.
        long_question = dict(question, text="paper " * 80 + "List two uses of this material.",
                             layer=layers - 1, max_new=0,
                             projection={"direction": u.tolist(), "strength": 1.0})
        projected = request("/api/agent/residual", long_question)
        assert 64 < projected["input_tokens"] <= 512
        stats = np.asarray(projected["intervened"]["prefill_writer_stats"])
        assert stats.shape == (1 + 2 * layers, 3)
        assert np.all(stats[:, 0] == projected["input_tokens"])
        assert float(stats[:, 1].sum()) > 0
        v = np.asarray(projected["intervened"]["residual"], dtype=np.float64)
        ratio = abs(float(v @ u)) / max(float(np.linalg.norm(v)), 1e-12)
        assert ratio < 5e-5, ratio
        assert projected["kl_baseline_to_intervened"] > 1e-8
        after = request("/api/agent/residual", question)
        assert after["baseline"]["token_ids"] == baseline["baseline"]["token_ids"]
        factors = request("/api/agent/residual/factors", {
            "model_key": key, "direction": u.tolist(),
            "direction_source": "Infrastructure test only: seeded random direction 261026; NOT a refusal direction",
            "destination": str((args.output / "random-control.npz").resolve()),
        })
        weight_zero = request("/api/agent/residual", dict(question, projection={
            "artifact": factors["path"], "strength": 0.0,
        }))
        assert weight_zero["intervened"]["token_ids"] == baseline["baseline"]["token_ids"]
        direct_partial = request("/api/agent/residual", dict(long_question, projection={
            "direction": u.tolist(), "strength": .25,
        }))
        weight_partial = request("/api/agent/residual", dict(long_question, projection={
            "artifact": factors["path"], "strength": .25,
        }))
        direct_vector = np.asarray(direct_partial["intervened"]["residual"])
        weight_vector = np.asarray(weight_partial["intervened"]["residual"])
        relative_error = float(np.linalg.norm(weight_vector - direct_vector) /
                               max(np.linalg.norm(direct_vector), 1e-12))
        # Predeclared coarse integration bound, NOT a bit-exact quantization claim.
        assert relative_error < .05, relative_error
        assert weight_partial["baseline_restoration_max_abs"] <= 1e-5
        report.update(status="passed", projected_component_ratio=ratio,
                      factor_vs_projection_residual_relative_error=relative_error,
                      interpretation="Callback coverage, zero identity and baseline recovery only; no refusal or capability gain claim")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["requests"] = records
        report["elapsed_seconds"] = time.monotonic() - started
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
