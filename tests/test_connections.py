"""Connection ranking, cached reuse and live graph routes without a global index."""

from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))
from connection_graph import rank_connections, walk_connections, reached_nodes


def test_rank_matches_layerwise_reference_and_zero_vectors():
    g = np.array([[1, 2, 7, 0], [0, 0, 0, 0], [-3, 2, 1, 0]], dtype=np.float32)
    u = np.array([[1, -2, 4, 0], [0, 0, 0, 0], [5, 2, 1, 0]], dtype=np.float32)
    out = rank_connections(g, u, 1, 2)
    assert [(c["layer"], c["n"], c["sign"]) for c in out["up"]] == [
        (0, 2, 1),
        (0, 1, -1),
    ]
    assert [(c["layer"], c["n"], c["sign"]) for c in out["down"]] == [
        (2, 0, -1),
        (2, 1, 1),
    ]
    for side in out.values():
        for c in side:
            x = np.abs(g[c["layer"]].astype(float) * u[c["layer"]])
            assert c["z"] == pytest.approx(
                (x[c["n"]] - x.mean()) / (x.std(ddof=1) + 1e-8), abs=5e-5
            )
    assert rank_connections(np.zeros((3, 4)), np.zeros((3, 4)), 1) == {
        "up": [],
        "down": [],
    }


def provider(l, n):
    return {
        "up": [{"layer": l - 1, "n": n, "z": 20, "sign": -1}] if l > 0 else [],
        "down": [{"layer": l + 1, "n": n, "z": 20, "sign": 1}] if l < 4 else [],
    }


def test_live_graph_reaches_both_directions_and_tracks_poles():
    graph = walk_connections(provider, 2, 0, first=1, k=1, hops=2)
    assert (
        len(graph["links"]) == 4
        and graph["expanded_neurons"] == 3
        and not graph["truncated"]
    )
    assert all(r[0] < r[2] for r in graph["links"])
    reach = reached_nodes(graph["links"], 2, 0)
    assert reach[(0, 0)][2] == 1 and reach[(1, 0)][2] == -1
    assert reach[(4, 0)][0] == 2


def test_live_graph_limits_are_explicit():
    graph = walk_connections(provider, 2, 0, first=1, k=1, hops=3, max_queries=1)
    assert (
        len(graph["links"]) == 2
        and graph["truncated"]
        and graph["expanded_neurons"] == 1
    )
    with pytest.raises(ValueError):
        walk_connections(provider, 2, 0, hops=4)
    with pytest.raises(ValueError, match="reversed"):
        walk_connections(
            lambda *a: {"up": [], "down": [{"layer": 1, "n": 0, "z": 20, "sign": 1}]},
            2,
            0,
        )


def test_runtime_cache_reuses_scores_across_k_and_label_requests(tmp_path, monkeypatch):
    from gguf_runtime import GGUFRuntime

    calls = []
    runtime = GGUFRuntime(str(tmp_path / "stats"), str(tmp_path / "cache"))
    monkeypatch.setattr(runtime, "ensure_loaded", lambda: None)

    def vectors(*coord):
        calls.append(coord)
        return np.ones((3, 4), dtype=np.float32), np.arange(
            12, dtype=np.float32
        ).reshape(3, 4)

    runtime.model = SimpleNamespace(connection_vectors=vectors)
    monkeypatch.setattr(
        runtime,
        "lens_many",
        lambda coords, k: {c: {"up": ["yes"], "down": ["no"]} for c in coords},
    )
    first = runtime.connections(1, 0, 2, 0)
    assert first["available"] and not first["cached"]
    first["down"][0]["z"] = -999
    second = runtime.connections(1, 0, 1, 1)
    assert (
        second["cached"]
        and second["down"][0]["z"] != -999
        and second["down"][0]["lens"] == ["yes"]
    )
    assert calls == [(1, 0)]
    assert runtime.cached_connections(0, 0)["pending"]


def test_web_and_agent_cone_work_without_precomputed_index(monkeypatch):
    from fastapi.testclient import TestClient
    import server

    monkeypatch.setattr(
        server.rt,
        "connection_graph",
        lambda l, n, first=5, k=2, hops=1, zmin=15, cap=500: walk_connections(
            provider, l, n, first, k, hops, zmin, cap
        ),
        raising=False,
    )
    monkeypatch.setattr(server.rt, "con_index", lambda: None)
    monkeypatch.setattr(
        server.rt,
        "lens_many",
        lambda coords, k: {c: {"up": ["yes"], "down": ["no"]} for c in coords},
        raising=False,
    )
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
        web = client.get("/api/web?layer=2&n=0&hops=2&first=1&k=1")
        assert web.status_code == 200 and web.headers["x-web-mode"] == "on-demand"
        assert np.frombuffer(web.content, np.int32).reshape(-1, 6).shape == (4, 6)
        cone = client.get("/api/agent/cone?layer=2&n=0&k=1&hops=2").json()
        assert (
            cone["links"] == 4 and cone["neurons"] == 4 and cone["mode"] == "on-demand"
        )
        assert all(c["steer_up"] == ["yes"] for c in cone["nodes"])


def test_cached_lens_is_readable_without_reloading_weights(tmp_path, monkeypatch):
    from gguf_runtime import GGUFRuntime

    runtime = GGUFRuntime(str(tmp_path / "stats"), str(tmp_path / "cache"))
    runtime._save("gguf-lens-f32-v1:14:8949", {"up": ["blue"], "down": ["red"]})

    def unavailable():
        raise AssertionError("A cached label must not load the model")

    monkeypatch.setattr(runtime, "ensure_loaded", unavailable)
    assert runtime.lens_many([(14, 8949)], 1) == {
        (14, 8949): {"up": ["blue"], "down": ["red"]}
    }
