import os
os.environ["EXPLORER_NO_MODEL"] = "1"
os.environ["EXPLORER_BUNDLED_CACHE"] = "1"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"explorer"))
from fastapi.testclient import TestClient
from server import app


def test_cache_exhibit_and_live_boundary():
    with TestClient(app, base_url="http://127.0.0.1:8791") as c:
        data = c.get("/api/index").json()
        assert data["bundled"] and data["cache_only"]
        assert data["model"] == "Qwen/Qwen3.5-4B"
        assert data["arch"]["inter"] == 9216
        r = c.post("/api/model/load", json={})
        assert r.status_code == 422 and "disabled" in r.json()["error"]
        assert c.get("/api/index", headers={"origin":"https://example.com"}).status_code == 403
        assert c.get("/api/index", headers={"host":"evil.example"}).status_code == 403


def test_reject_invalid_coordinates_paths_and_provider():
    with TestClient(app, base_url="http://127.0.0.1:8791") as c:
        assert c.post("/api/name", json={"layer":-1,"n":0,"name":"invalid"}).status_code == 422
        assert c.get("/api/lens?layer=0&n=-1").status_code == 422
        assert c.get("/api/layout?kind=../../config").status_code == 422
        assert c.post("/api/agent/chat", json={"provider":"unknown","message":"hello"}).status_code == 422
        assert c.post("/api/agent/chat", json={"chat":"../../private","message":"hello"}).status_code == 422


def test_unavailable_connections_remain_explicit(monkeypatch):
    from server import rt
    monkeypatch.setattr(rt, "connections", lambda *args: {"up": [], "down": [], "available": False, "reason": "Unsupported backend operation"})
    with TestClient(app, base_url="http://127.0.0.1:8791") as c:
        out = c.get("/api/connections?layer=0&n=0").json()
        assert out["available"] is False
        assert out["reason"] == "Unsupported backend operation"
