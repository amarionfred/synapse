"""Ranking and bounded traversal shared by live connection providers."""

from collections import deque
import numpy as np


def rank_connections(gate, up, layer, k=64):
    """Layerwise z of |gate alignment * up alignment|, as in the original explorer."""
    if gate.shape != up.shape or gate.ndim != 2 or not 0 <= layer < len(gate):
        raise ValueError("Invalid connection vectors")
    if not np.isfinite(gate).all() or not np.isfinite(up).all():
        raise ValueError("Nonfinite connection vectors")
    product = gate.astype(np.float64) * up.astype(np.float64)
    magnitude = np.abs(product)
    spread = (
        magnitude.std(axis=1, ddof=1, keepdims=True)
        if gate.shape[1] > 1
        else np.zeros((len(gate), 1))
    )
    z = (magnitude - magnitude.mean(axis=1, keepdims=True)) / (spread + 1e-8)
    out = {"up": [], "down": []}
    for side, layers in (("up", range(layer)), ("down", range(layer + 1, len(gate)))):
        for l in layers:
            # Stable ordering makes ties reproducible. A zero vector has no links.
            order = np.argsort(-z[l], kind="stable")[:k]
            out[side].extend(
                {
                    "layer": l,
                    "n": int(n),
                    "z": round(float(z[l, n]), 4),
                    "sign": 1 if product[l, n] > 0 else -1,
                }
                for n in order
                if magnitude[l, n] > 0
            )
        out[side].sort(key=lambda c: (-c["z"], c["layer"], c["n"]))
        out[side] = out[side][:k]
    return out


def walk_connections(
    fetch, layer, n, first=5, k=2, hops=1, zmin=15, cap=500, max_queries=12
):
    """Fetch each expansion once. Links always point from earlier to later layers."""
    if not (1 <= first <= 64 and 1 <= k <= 64 and 1 <= hops <= 3 and 2 <= cap <= 500):
        raise ValueError("Use 1..64 links, 1..3 hops and a 2..500 edge cap")
    root = (layer, n)
    queue = deque([("down", root, 1), ("up", root, 1)])
    seen = {"down": {root}, "up": {root}}
    fetched, rows, counts = {}, [], {"down": 0, "up": 0}
    truncated = False
    while queue:
        side, node, hop = queue.popleft()
        if counts[side] >= cap // 2:
            truncated = True
            continue
        if node not in fetched:
            if len(fetched) >= max_queries:
                truncated = True
                continue
            fetched[node] = fetch(*node)
        edges = fetched[node][side][: first if hop == 1 else k]
        for edge in edges:
            if hop > 1 and edge["z"] < zmin:
                continue
            target = (edge["layer"], edge["n"])
            src, dst = (node, target) if side == "down" else (target, node)
            if src[0] >= dst[0]:
                raise ValueError(
                    "A connection provider returned a reversed or same-layer edge"
                )
            if counts[side] >= cap // 2:
                truncated = True
                break
            rows.append(
                (*src, *dst, edge["z"] * edge["sign"], hop if side == "down" else -hop)
            )
            counts[side] += 1
            if hop < hops and target not in seen[side]:
                queue.append((side, target, hop + 1))
                seen[side].add(target)
    return {
        "links": rows,
        "hops": hops,
        "expanded_neurons": len(fetched),
        "truncated": truncated,
        "mode": "on-demand",
        "max_queries": max_queries,
    }


def reached_nodes(links, layer, n):
    """First BFS route supplies a node's pole; merged paths need causal tests."""
    signs, reach = {(layer, n): 1}, {}
    for sl, sn, dl, dn, z, hop in links:
        near, far = ((sl, sn), (dl, dn)) if hop > 0 else ((dl, dn), (sl, sn))
        if near in signs and far not in signs:
            signs[far] = signs[near] * (1 if z > 0 else -1)
            reach[far] = (hop, abs(z), signs[far])
    return reach
