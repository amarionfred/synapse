"""Direct-path connections for every neuron in every layer, for the explorer's details tab.

For layers i < j, P[B, A] = (W_gate_j[B] . v_A) * (W_up_j[B] . v_A) with v_A = gain_j * W_down_i[:, A], where gain_j is the adapter-validated
effective pre-MLP RMSNorm gain. Neuron A's downstream links in layer j are column A of P, |P| z-scored over B; neuron
B's upstream links in layer i are row B of P, |P| z-scored over A. Each neuron keeps its k strongest links per side
across all layers, with sign(P).

    uv run python explorer/build_connections.py            # CUDA, about 5 min on a 4090

Writes the selected checkpoint cache connections.npz: {down,up}_layer [L, N, k] int32, {down,up}_n int32, {down,up}_z float16,
{down,up}_sign int8.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.getenv("EXPLORER_MODEL", "Qwen/Qwen3.5-4B"))
ap.add_argument("--revision", default=os.getenv("EXPLORER_REVISION", "main"))
ap.add_argument("--k", type=int, default=24)
ap.add_argument("--out", default=None)
a = ap.parse_args()
os.environ["EXPLORER_MODEL"] = a.model
os.environ["EXPLORER_REVISION"] = a.revision
from model_config import CACHE, STATE
from runtime import Runtime
rt = Runtime(str(STATE / "stats"))
rt.ensure_loaded()
if not rt.adapter.connections_supported:
    raise SystemExit("This architecture does not support the gated-MLP/RMSNorm connection heuristic")
a.out = a.out or str(CACHE / "connections.npz")
dev = rt.dev
L, N = len(rt.adapter.layers), rt.adapter.down[0].width
K = min(a.k, N)
if K < 1:
    raise SystemExit("k must be positive")
down = [p.weight.detach() for p in rt.adapter.down]
gate = [l.mlp.gate_proj.weight.detach() for l in rt.adapter.layers]
up = [l.mlp.up_proj.weight.detach() for l in rt.adapter.layers]
norm = [rt.adapter.norm_gain(l).to(down[l].dtype) for l in range(L)]

# running top-k per neuron and side: score, source layer, source neuron, sign
best = {s: (torch.full((L, N, K), -1e9, device=dev), torch.zeros((L, N, K), dtype=torch.int32, device=dev),
            torch.zeros((L, N, K), dtype=torch.int32, device=dev), torch.zeros((L, N, K), dtype=torch.int8, device=dev))
        for s in ("down", "up")}


def merge(side, l, z, idx, other_layer, sign):
    bz, bl, bn, bs = best[side]
    cz = torch.cat([bz[l], z], 1)
    cl = torch.cat([bl[l], torch.full_like(idx, other_layer, dtype=torch.int32)], 1)
    cn = torch.cat([bn[l], idx.to(torch.int32)], 1)
    cs = torch.cat([bs[l], sign], 1)
    top = cz.topk(K, dim=1)
    bz[l], bl[l], bn[l], bs[l] = top.values, cl.gather(1, top.indices), cn.gather(1, top.indices), cs.gather(1, top.indices)


t0 = time.time()
with torch.inference_mode():
    for j in range(1, L):
        for i in range(j):
            V = down[i] * norm[j][:, None]  # [hidden, N_i]
            P = (gate[j] @ V).float() * (up[j] @ V).float()  # [N_j (B), N_i (A)]
            M = P.abs()
            # downstream of A (layer i): column A over B
            zc = (M - M.mean(0, keepdim=True)) / (M.std(0, keepdim=True) + 1e-8)
            top = zc.topk(K, dim=0)  # [K, N_i]
            sgn = torch.where(P.gather(0, top.indices) > 0, 1, -1).to(torch.int8)
            merge("down", i, top.values.T, top.indices.T, j, sgn.T)
            # upstream of B (layer j): row B over A
            zr = (M - M.mean(1, keepdim=True)) / (M.std(1, keepdim=True) + 1e-8)
            top = zr.topk(K, dim=1)  # [N_j, K]
            sgn = torch.where(P.gather(1, top.indices) > 0, 1, -1).to(torch.int8)
            merge("up", j, top.values, top.indices, i, sgn)
            del P, M, zc, zr
        print(f"L{j} done ({time.time() - t0:.0f}s)", flush=True)

out = {}
for side, (bz, bl, bn, bs) in best.items():
    out[f"{side}_z"] = bz.clamp(min=-60000).cpu().numpy().astype(np.float16)
    out[f"{side}_layer"] = bl.cpu().numpy().astype(np.int32)
    out[f"{side}_n"] = bn.cpu().numpy()
    out[f"{side}_sign"] = bs.cpu().numpy()
np.savez_compressed(a.out, **out, k=K)
print("->", a.out, f"{os.path.getsize(a.out) / 1e6:.0f} MB")
