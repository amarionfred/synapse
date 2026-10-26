"""Native Qwen3.5 GGUF weights; generation is a separate, admitted resource."""

import json
from pathlib import Path
import subprocess
import time
from collections import Counter
import numpy as np
import torch
from runtime import Runtime, REFERENCE_TEXT
from model_config import MODEL, ARCH, NO_MODEL
from llama_bridge import Bridge, Tokenizer
from connection_graph import rank_connections, walk_connections, reached_nodes


class GGUFRuntime(Runtime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inspector = None

    def inspection_bridge(self):
        """Called under self.lock. Never loads inference or takes its CPU slot."""
        if self.model is not None:
            return self.model
        if self.inspector is None:
            torch.set_num_threads(4)
            self.inspector = Bridge(str(Path(MODEL).expanduser().resolve()), weights_only=True)
        return self.inspector

    def ensure_loaded(self):
        with self.lock:
            if self.model is not None:
                return
            if NO_MODEL:
                raise RuntimeError("Live model disabled")
            if (
                subprocess.run(
                    [
                        "systemctl",
                        "--user",
                        "is-active",
                        "--quiet",
                        "qwen-openweights-lab.service",
                    ],
                    capture_output=True,
                ).returncode
                == 0
            ):
                raise RuntimeError(
                    "Stop the separate Qwen chat runtime first: qwen-lab stop. Both use the same memory budget."
                )
            mem = dict(
                line.split(":", 1)
                for line in Path("/proc/meminfo").read_text().splitlines()
            )
            available = int(mem["MemAvailable"].split()[0]) * 1024
            # The real connection/lens check peaked at 3.7 GiB for the whole
            # process. Imports already occupy ~0.45 GiB here; 4.25 GiB available
            # leaves about 1 GiB beyond that measured incremental requirement.
            required = max(
                int(4.25 * 2**30), Path(MODEL).stat().st_size + int(1.5 * 2**30)
            )
            if available < required:
                raise RuntimeError(
                    f"Qwen GGUF needs {(required + 2**20 - 1) // 2**20:,} MiB available before loading; {available // 2**20:,} MiB available. Close an unused app and retry."
                )
            self.status = "loading GGUF"
            bridge = None
            try:
                torch.set_num_threads(4)
                if self.inspector is not None:
                    self.inspector.close()
                    self.inspector = None
                bridge = Bridge(str(Path(MODEL).expanduser().resolve()))
                # Prove every layer's exact tensor name, type and shape before advertising readiness.
                bridge.run(bridge.tokenize("Hello"), observe=True, residual=True)
                self.model, self.tok, self.dev = bridge, Tokenizer(bridge), "cpu"
                self.status = "ready"
            except Exception as e:
                if bridge:
                    bridge.close()
                self.status = f"error: {e}"
                raise

    def capabilities(self):
        return {
            "backend": "llama.cpp b11669 (CPU GGUF)",
            "adapter": "qwen35 ffn_swiglu → ffn_down",
            "validated": self.model is not None,
            "inference_loaded": self.model is not None,
            "inspection_loaded": self.model is not None or self.inspector is not None,
            "inspection_mode": "read-only mmap + vocabulary; no inference context",
            "layers": ARCH["n_layers"],
            "neurons_per_layer": ARCH["inter"],
            "capabilities": {
                "activations": True,
                "additive_steering": True,
                "multiplicative_steering": True,
                "logit_lens": True,
                "weight_connections": True,
                "attention_heads": [0] * ARCH["n_layers"],
            },
            "dose_units": "median reference residual norm / dequantized output-projection column norm; quantized matmul adds numerical error",
            "context_tokens": 2048,
            "connection_mode": "on-demand",
            "connection_hops": {
                "default": 1,
                "maximum": 3,
                "max_expansions_per_request": 12,
            },
            "generation": "three independent greedy rows, sequential on CPU",
            "note": "Links and labels use this GGUF's read-only weights without inference loading; hover is cache-only. Activations and steering still require admitted inference. Head ablation and whole-model lens indexing are not implemented. Quantized calibration is separate from the author's data.",
        }

    def n_heads(self, layer):
        self.validate_coordinate(layer)
        return 0

    def _check_heads(self, heads):
        if heads:
            raise ValueError("GGUF attention-head interventions are not implemented")

    def _ids(self, text, chat):
        ids = self.chat_ids(text) if chat else self.tok(text).input_ids
        if not 1 <= len(ids) <= 2048:
            raise ValueError("GGUF probes require 1..2048 input tokens")
        return ids

    def down_col(self, layer, n):
        self.validate_coordinate(layer, n)
        with self.lock:
            return torch.from_numpy(self.inspection_bridge().column(layer, n))

    def _layer_resid_p50(self, layer):
        self.validate_coordinate(layer)
        if layer not in self.resid_p50:
            self.ensure_loaded()
            path = Path(self.cache_dir) / "gguf_residual_norms.json"
            if path.exists():
                self.resid_p50.update(
                    {int(k): float(v) for k, v in json.loads(path.read_text()).items()}
                )
            else:
                ids = self.tok(REFERENCE_TEXT).input_ids[:128]
                _, _, res = self.model.run(ids, residual=True)
                # Same lower-median convention as torch.median in the Transformers backend.
                self.resid_p50 = {
                    i: float(torch.from_numpy(r[1:]).median())
                    for i, r in enumerate(res)
                }
                path.write_text(json.dumps(self.resid_p50, indent=2) + "\n")
        return self.resid_p50[layer]

    def _pushes(self, pushes, sign=1):
        out = []
        for l, n, dose, mode in pushes or []:
            self.validate_coordinate(l, n)
            if mode not in {"add", "mul"} or not np.isfinite(dose):
                raise ValueError("Use a finite dose and mode add or mul")
            if n is None and mode != "mul":
                raise ValueError("Whole-layer interventions must multiply")
            if mode == "mul":
                value = 1.0 if sign == 0 else sign * dose
            elif sign == 0 or dose == 0:
                value = 0.0
            else:
                norm = float(self.down_col(l, n).norm())
                if norm <= 1e-12:
                    raise ValueError("Cannot normalize a zero write vector")
                value = sign * dose * self.layer_resid_p50(l) / norm
            if not np.isfinite(np.float32(value)):
                raise ValueError("Intervention exceeds float32 range")
            out.append((l, n, mode, value))
        return out

    def read(self, layer, n, text, chat=True, pushes=None, heads=None):
        self.validate_coordinate(layer, n)
        self._check_heads(heads)
        self.ensure_loaded()
        with self.lock:
            ids = self._ids(text, chat)
            _, act, _ = self.model.run(
                ids, read=(layer, n), pushes=self._pushes(pushes)
            )
            units = []
            for text, idx in self.token_units(ids):
                vals = [float(act[i]) for i in idx]
                units.append({"text": text, "act": max(vals, key=abs), "pieces": vals})
            return {"units": units}

    def full_activations(self, ids):
        self.ensure_loaded()
        if len(ids) * ARCH["n_layers"] * ARCH["inter"] * 4 > 160 << 20:
            raise ValueError(
                "Full GGUF activation capture exceeds the 160 MiB probe budget"
            )
        with self.lock:
            return self.model.run(ids, full=True)[1]

    def all_activations(self, ids, skip_first=False):
        a = self.full_activations(ids)[:, int(skip_first) :, :]
        return {
            l: (
                torch.from_numpy(v.max(0)),
                torch.from_numpy(v.argmax(0)),
                torch.from_numpy(v.min(0)),
                torch.from_numpy(v.argmin(0)),
            )
            for l, v in enumerate(a)
        }

    def next_logprobs(self, text, chat=True, prefix="", pushes=None, heads=None):
        self._check_heads(heads)
        self.ensure_loaded()
        with self.lock:
            ids = (
                self._ids(text, chat)
                + self.tok(prefix, add_special_tokens=False).input_ids
                if prefix
                else self._ids(text, chat)
            )
            base = torch.log_softmax(torch.from_numpy(self.model.run(ids)[0]), -1)
            changed = (
                torch.log_softmax(
                    torch.from_numpy(
                        self.model.run(ids, pushes=self._pushes(pushes))[0]
                    ),
                    -1,
                )
                if pushes
                else None
            )
            return base, changed

    def steer_multi(self, pushes, rows_messages, max_new=200, cancel=None):
        if len(rows_messages) != 3 or not 1 <= max_new <= 1024:
            raise ValueError(
                "Steering needs three conversations and 1..1024 new tokens"
            )
        self.ensure_loaded()
        with self.lock:
            seqs = [self.messages_ids(m) for m in rows_messages]
            if any(not q or len(q) + max_new > 2048 for q in seqs):
                raise ValueError(
                    "Prompt plus generation exceeds this GGUF runtime's 2048-token context"
                )
            for row, (ids, sign) in enumerate(zip(seqs, (1, 0, -1))):
                if cancel is not None and cancel.is_set():
                    break
                ps = self._pushes(pushes, sign)
                logits = self.model.run(ids, pushes=ps)[0]
                generated, sent = [], ""
                for step in range(max_new):
                    if cancel is not None and cancel.is_set():
                        break
                    token = int(logits.argmax())
                    if self.model.lib.syn_eog(self.model.handle, token):
                        break
                    generated.append(token)
                    text = self.tok.decode(generated, skip_special_tokens=True)
                    if not text.endswith("\ufffd"):
                        yield row, text[len(sent) :]
                        sent = text
                    if step + 1 < max_new:
                        logits = self.model.run([token], reset=False, pushes=ps)[0]
        yield None, None

    def lens(self, layer, n, k=16):
        return self.lens_many([(layer, n)], k)[(layer, n)]

    def _cached(self, key):
        with self._db_lock:
            row = self._db.execute(
                "SELECT val FROM con WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def _save(self, key, value):
        with self._db_lock:
            self._db.execute(
                "INSERT OR REPLACE INTO con VALUES (?, ?)", (key, json.dumps(value))
            )
            self._db.commit()

    def lens_many(self, coordinates, k=16):
        coords = list(dict.fromkeys(coordinates))
        if not 1 <= k <= 32 or len(coords) > 128:
            raise ValueError("Request at most 128 lens directions and 1..32 tokens")
        for l, n in coords:
            self.validate_coordinate(l, n)
        if not coords:
            return {}
        with self.lock, torch.inference_mode():
            result, missing = {}, []
            for coord in coords:
                cache_key = f"gguf-lens-f32-v1:{coord[0]}:{coord[1]}"
                row = self._lens_cache.get(coord) or self._cached(cache_key)
                if row is None:
                    missing.append(coord)
                else:
                    result[coord] = row
            if missing:
                weights = self.inspection_bridge()
                tok = Tokenizer(weights)
                # Stream the unembedding once for a batch of directions. No full
                # FP32 checkpoint or whole-model lens index is allocated.
                directions = torch.from_numpy(
                    np.stack([weights.column(*c) for c in missing], axis=1)
                )
                gain = torch.from_numpy(
                    weights.weight_rows("output_norm.weight", 0, 1)[0]
                )
                scales = (
                    (directions.double().square().mean(0) + weights.rms_epsilon)
                    .rsqrt()
                    .float()
                )
                directions = directions * (gain[:, None] * scales)
                best = {
                    side: (
                        torch.empty((0, len(missing))),
                        torch.empty((0, len(missing)), dtype=torch.int64),
                    )
                    for side in ("up", "down")
                }
                for start in range(0, weights.vocab, 1024):
                    rows = torch.from_numpy(
                        weights.weight_rows(
                            "output.weight", start, min(1024, weights.vocab - start)
                        )
                    )
                    logits = rows @ directions
                    for side, sign in (("up", 1), ("down", -1)):
                        val, idx = (sign * logits).topk(min(32, len(logits)), dim=0)
                        old_val, old_idx = best[side]
                        vals, ids = (
                            torch.cat((old_val, val)),
                            torch.cat((old_idx, idx + start)),
                        )
                        val, order = vals.topk(min(32, len(vals)), dim=0)
                        best[side] = (val, ids.gather(0, order))
                for i, coord in enumerate(missing):
                    row = {
                        side: [
                            tok.decode([t]) for t in best[side][1][:, i].tolist()
                        ]
                        for side in ("up", "down")
                    }
                    result[coord] = row
                    self._save(f"gguf-lens-f32-v1:{coord[0]}:{coord[1]}", row)
            if len(self._lens_cache) > 2048:
                self._lens_cache.clear()
            self._lens_cache.update(result)
            return {
                coord: {side: row[side][:k] for side in ("up", "down")}
                for coord, row in result.items()
            }

    def connections(self, layer, n, k=24, lens_k=5):
        self.validate_coordinate(layer, n)
        if not 1 <= k <= 64 or not 0 <= lens_k <= 32:
            raise ValueError("Request 1..64 connections and 0..32 lens tokens")
        key = f"gguf-links-v1:{layer}:{n}"
        with self.lock:
            data = self._cached(key)
            cached = data is not None
            if data is None:
                weights = self.inspection_bridge()
                started = time.monotonic()
                gate, up = weights.connection_vectors(layer, n)
                data = rank_connections(gate, up, layer)
                data.update(
                    available=True,
                    method="gated-MLP direct-path weight alignment",
                    mode="on-demand",
                    scope="all other layers of this quantized checkpoint",
                    compute_seconds=time.monotonic() - started,
                )
                self._save(key, data)
            out = {
                **data,
                "cached": cached,
                **{side: [dict(c) for c in data[side][:k]] for side in ("up", "down")},
            }
            if lens_k:
                neighbors = out["up"] + out["down"]
                lenses = self.lens_many(
                    [(c["layer"], c["n"]) for c in neighbors], lens_k
                )
                for c in neighbors:
                    label = lenses[(c["layer"], c["n"])]
                    c.update(lens=label["up"], lens_neg=label["down"])
            return out

    def cached_lens(self, layer, n, k=8):
        self.validate_coordinate(layer, n)
        row = self._lens_cache.get((layer, n)) or self._cached(f"gguf-lens-f32-v1:{layer}:{n}")
        return ({side: row[side][:k] for side in ("up", "down")} if row is not None
                else {"up": [], "down": [], "pending": True})

    def cached_connections(self, layer, n, k=16):
        self.validate_coordinate(layer, n)
        data = self._cached(f"gguf-links-v1:{layer}:{n}")
        if data is None:
            return {"up": [], "down": [], "available": True, "pending": True}
        return {**data, **{side: data[side][:k] for side in ("up", "down")}}

    def connection_graph(self, layer, n, first=5, k=2, hops=1, zmin=15, cap=500):
        self.validate_coordinate(layer, n)
        return walk_connections(
            lambda l, m: self.connections(l, m, 64, 0),
            layer,
            n,
            first,
            k,
            hops,
            zmin,
            cap,
        )

    def connection_summary(self, layer, n, first=5, k=2, hops=1, top=24):
        graph = self.connection_graph(layer, n, first, k, hops)
        reach = reached_nodes(graph["links"], layer, n)
        # Label the strongest 64 reached nodes; report the scope rather than
        # inventing corpus-frequency/layerwise-lift statistics without an index.
        selected = sorted(reach, key=lambda c: (abs(reach[c][0]), -reach[c][1], c))[:64]
        lens = self.lens_many(selected, 8)
        zmax = max((v[1] for v in reach.values()), default=1.0) or 1.0

        def pool(nodes, flip):
            counts, weights = Counter(), Counter()
            for c in nodes:
                hop, z, sign = reach[c]
                tokens = {
                    t.strip()
                    for t in lens[c]["up" if sign * flip > 0 else "down"]
                    if t.strip() and "\ufffd" not in t
                }
                for token in tokens:
                    counts[token] += 1
                    weights[token] += z / zmax * 0.8 ** (abs(hop) - 1)
            return [
                {"tok": t, "n": counts[t], "w": round(weights[t], 3)}
                for t in sorted(weights, key=lambda t: (-weights[t], t))[:top]
            ]

        up, down = (
            [c for c in selected if reach[c][0] < 0],
            [c for c in selected if reach[c][0] > 0],
        )
        return {
            "src": "on-demand logit",
            "ranking": "weighted token counts among shown neighbors",
            "links": len(graph["links"]),
            "hops": hops,
            "truncated": graph["truncated"],
            "expanded_neurons": graph["expanded_neurons"],
            "labelled_neurons": len(selected),
            "reached_neurons": len(reach),
            "up": {"neurons": len(up), "fed_by": pool(up, 1), "held_by": pool(up, -1)},
            "down": {
                "neurons": len(down),
                "toward": pool(down, 1),
                "away": pool(down, -1),
            },
        }

    def vocab_strings(self):
        self.ensure_loaded()
        with self.lock:
            if getattr(self, "_vocab", None) is None:
                self._vocab = [self.tok.decode([i]) for i in range(len(self.tok))]
            return self._vocab

    def build_lens_index(self, *args, **kwargs):
        raise ValueError(
            "Full GGUF lens indexing is not implemented; per-neuron lens works on demand"
        )

    def build_act_scale(self, *args, **kwargs):
        raise ValueError(
            "GGUF uses the bounded fixed-reference calibration; corpus sweeps are not implemented"
        )
