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


def test_cached_geometry_does_not_wait_for_slow_label_inspection(tmp_path, monkeypatch):
    import server
    fake = SimpleNamespace(cached_connections=lambda *args: {
        "up": [{"layer": 0, "n": 3, "z": 20, "sign": 1}],
        "down": [{"layer": 31, "n": 4, "z": 21, "sign": -1}], "available": True})
    def forbidden(*args, **kwargs):
        raise AssertionError("Cached geometry tried to join the slow native/label queue")
    fake.connections = forbidden
    monkeypatch.setattr(server, "rt", fake)
    monkeypatch.setattr(server, "GGUF", True)
    async def scenario():
        gate = asyncio.Lock()
        await gate.acquire()  # An older selection is still computing labels.
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(inspection_gate=gate)))
        result = await asyncio.wait_for(server.connections(29, 6872, 24, 0, request), .5)
        assert len(result["up"]) == len(result["down"]) == 1
        assert gate.locked()
        gate.release()
    asyncio.run(scenario())


def test_canceled_label_worker_releases_slot_only_after_cleanup():
    from server import inspection_call
    async def scenario():
        gate = asyncio.Lock()
        entered, canceled, release = threading.Event(), threading.Event(), threading.Event()
        async def connected(): return False
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(inspection_gate=gate)), is_disconnected=connected)
        def labels(*, cancel):
            entered.set()
            assert cancel.wait(2)
            canceled.set()
            assert release.wait(2)
        task = asyncio.create_task(inspection_call(request, labels, cancelable=True))
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass
        assert await asyncio.to_thread(canceled.wait, 1)
        assert gate.locked()
        next_request = asyncio.create_task(inspection_call(request, lambda: "next selection"))
        await asyncio.sleep(.01)
        assert not next_request.done()
        release.set()
        assert await asyncio.wait_for(next_request, 2) == "next selection"
        assert not gate.locked()
    asyncio.run(scenario())


def test_label_cancellation_stops_before_more_output_rows(tmp_path, monkeypatch):
    import pytest
    from gguf_runtime import GGUFRuntime
    rt = GGUFRuntime(str(tmp_path / 'stats'), str(tmp_path / 'cache'))
    cancel = threading.Event()
    rows = []
    def weight_rows(name, start, count):
        if name == 'output.weight':
            rows.append(start)
            cancel.set()  # A selection changed during this bounded native chunk.
        return np.ones((count, 3), np.float32)
    weights = SimpleNamespace(vocab=2050, chat_template=None, rms_epsilon=1e-6,
                              column=lambda *args: np.ones(3, np.float32), weight_rows=weight_rows)
    monkeypatch.setattr(rt, 'inspection_bridge', lambda: weights)
    monkeypatch.setattr(rt, 'ensure_loaded', lambda: (_ for _ in ()).throw(AssertionError('Loaded inference')))
    with pytest.raises(RuntimeError, match='Inspection canceled'):
        rt.lens_many([(0, 3)], 8, cancel=cancel)
    assert rows == [0]  # No second chunk, no incomplete cache entry.
    assert rt.cached_lens(0, 3)['pending']
    assert rt.model is None
    assert rt.lock.acquire(blocking=False)
    rt.lock.release()
