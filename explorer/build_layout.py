"""Seriated sheet layouts: places each layer's neurons on its ceil(sqrt(N))-wide grid so that neurons with similar weights sit
together. Embeds each neuron by its down_proj column ("writes": the direction it adds to the residual stream) or its
gate and up rows ("reads"), cosine-normalised, PCA to 50 dims, UMAP to 2D, then snaps to the grid: sort by y into
rows of ceil(sqrt(N)), then by x within each row.

    uv run python explorer/build_layout.py --kind writes

Writes the checkpoint cache layout_<kind>.npz: cell[L, N] (int32 grid cell for neuron n of layer L).
"""
import argparse
import json
import os

import numpy as np
import umap

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.getenv("EXPLORER_MODEL", "Qwen/Qwen3.5-4B"))
ap.add_argument("--revision", default=os.getenv("EXPLORER_REVISION", "main"))
ap.add_argument("--kind", default="writes", choices=["writes", "reads"])
ap.add_argument("--pca", type=int, default=50)
ap.add_argument("--neighbors", type=int, default=30)
a = ap.parse_args()

os.environ["EXPLORER_MODEL"] = a.model
os.environ["EXPLORER_REVISION"] = a.revision
from model_config import CACHE, STATE
from runtime import Runtime
from adapters import first, Projection
rt = Runtime(str(STATE / "stats"))
rt.ensure_loaded()
L, N = len(rt.adapter.layers), rt.adapter.down[0].width
side = int(np.ceil(N ** 0.5))
cells = np.zeros((L, N), np.int32)
for l, layer in enumerate(rt.adapter.layers):
    if a.kind == "writes":
        X = rt.adapter.down[l].weight.detach().float().cpu().numpy().T
    elif hasattr(layer, "mlp") and hasattr(layer.mlp, "gate_proj"):
        X = np.concatenate([layer.mlp.gate_proj.weight.detach().float().cpu().numpy(),
                            layer.mlp.up_proj.weight.detach().float().cpu().numpy()], 1)
    else:
        path, module = first(layer, ["mlp.c_fc", "mlp.dense_h_to_4h", "fc1", "mlp.fc1"])
        if module is None:
            raise SystemExit("This architecture needs a reader-projection layout adapter; writes layout is available")
        X = Projection.wrap(module, path).weight.detach().float().cpu().numpy()
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    X = X - X.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    Z = U[:, : a.pca] * S[: a.pca]
    xy = umap.UMAP(n_neighbors=a.neighbors, min_dist=0.05, metric="cosine", random_state=0).fit_transform(Z)
    by_y = np.argsort(xy[:, 1], kind="stable")
    for r in range(side):
        band = by_y[r * side:(r + 1) * side]
        band = band[np.argsort(xy[band, 0], kind="stable")]
        cells[l, band] = r * side + np.arange(len(band))
    print(f"L{l} done", flush=True)

out = os.path.join(CACHE, f"layout_{a.kind}.npz")
np.savez(out, cell=cells)
print("->", out)
