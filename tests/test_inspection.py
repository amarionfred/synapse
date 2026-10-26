"""Cold inspection and hover must not need a generation model or queued scans."""
from pathlib import Path
import sys
from types import SimpleNamespace
import asyncio
import threading

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))


def test_cold_connections_use_weights_without_inference(tmp_path, monkeypatch):
    from gguf_runtime import GGUFRuntime

    rt = GGUFRuntime(str(tmp_path / "stats"), str(tmp_path / "cache"))

    def forbidden():
        raise AssertionError("Inspection attempted to load inference")

    weights = SimpleNamespace(connection_vectors=lambda *a: (
        np.ones((3, 4), np.float32), np.arange(12, dtype=np.float32).reshape(3, 4)))
    monkeypatch.setattr(rt, "ensure_loaded", forbidden)
    monkeypatch.setattr(rt, "inspection_bridge", lambda: weights, raising=False)
    out = rt.connections(1, 0, 2, 0)
    assert out["up"] and out["down"] and not out["cached"]
    assert rt.model is None and rt.tok is None and rt.status == "idle"


def test_hover_never_computes_missing_weights(monkeypatch):
    from fastapi.testclient import TestClient
    import server

    def forbidden(*a):
        raise AssertionError("Hover must not start a weight computation")

    fake = SimpleNamespace(
        cached_connections=lambda *a: {"up": [], "down": [], "pending": True},
        cached_lens=lambda *a: {"up": [], "down": [], "pending": True},
        connections=forbidden, lens=forbidden)
    monkeypatch.setattr(server, "rt", fake)
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
        r = client.get("/api/hover?layer=31&n=7066")
    assert r.status_code == 200
    assert r.json()["pending"] and r.json()["lens"] == []


def test_abandoned_requests_skip_scans_and_running_scan_holds_gate():
    from fastapi import HTTPException
    from server import inspection_call

    async def scenario():
        gate = asyncio.Lock()
        state = SimpleNamespace(inspection_gate=gate)
        started, release = threading.Event(), threading.Event()
        calls = []

        def request(disconnected=False):
            async def is_disconnected():
                return disconnected
            return SimpleNamespace(app=SimpleNamespace(state=state), is_disconnected=is_disconnected)

        def scan():
            started.set()
            assert release.wait(5)
            calls.append("running")

        first = asyncio.create_task(inspection_call(request(), scan))
        assert await asyncio.to_thread(started.wait, 2)
        first.cancel()
        try:
            await first
        except asyncio.CancelledError:
            pass
        assert gate.locked()
        stale = asyncio.create_task(inspection_call(request(True), lambda: calls.append("stale")))
        last = asyncio.create_task(inspection_call(request(), lambda: calls.append("latest")))
        release.set()
        try:
            await stale
        except HTTPException as e:
            assert e.status_code == 499
        else:
            raise AssertionError("Disconnected selection was computed")
        await last
        assert calls == ["running", "latest"]
        assert not gate.locked()

    asyncio.run(scenario())
