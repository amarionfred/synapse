"""Checkpoint binding, malformed requests, direction estimation and factor orientation."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))
from residual_projection import (
    bounded_strength, contrastive_direction, export_factors,
    projection_request, unit_direction, load_factors,
)


def test_contrast_requires_matched_development_pairs():
    answered = np.array([[1, 2, 3], [3, 1, 4], [2, 0, 1]], float)
    refused = answered + [0, 4, -3]
    u = contrastive_direction(refused, answered)
    np.testing.assert_allclose(u, [0, .8, -.6], atol=1e-7)
    np.testing.assert_allclose(contrastive_direction(answered, refused), -u)
    for r, a in [(refused[:2], answered[:2]), (refused, answered[:2]),
                 (answered, answered), (refused * np.nan, answered)]:
        with pytest.raises(ValueError):
            contrastive_direction(r, a)


def test_request_rejects_cross_checkpoint_and_nonunit_direction():
    body = {"model_key": "correct", "text": "Hello", "layer": 1,
            "max_new": 0, "projection": {"direction": [1, 0, 0], "strength": .25}}
    text, layer, max_new, u, strength = projection_request(body, "correct", 3, 2)
    assert (text, layer, max_new, strength) == ("Hello", 1, 0, .25)
    assert u.dtype == np.float32
    for change in [{"model_key": "other"}, {"layer": -1}, {"layer": True},
                   {"max_new": 129}, {"max_new": 1.5}, {"text": ""},
                   {"projection": {"direction": [2, 0, 0], "strength": .5}}]:
        with pytest.raises(ValueError):
            projection_request(dict(body, **change), "correct", 3, 2)
    for v in [float("nan"), float("inf"), 1.01, True, ".5", 10**400]:
        with pytest.raises(ValueError):
            bounded_strength(v)
    for v in [[0, 0, 0], [1, 0], [[1, 0, 0]], [float("inf"), 0, 0], {"x": 1}]:
        with pytest.raises(ValueError):
            unit_direction(v, 3)


def test_factor_export_matches_explicit_projection_and_preserves_weights(tmp_path, monkeypatch):
    class FakeBridge:
        model_key = "fixture-only"
        identity = {"model": "synthetic", "schema": 2}
        arch = {"hidden": 3, "n_layers": 2, "model_type": "qwen35",
                "layer_types": ["linear_attention", "full_attention"]}
        weights = {
            "blk.0.ssm_out.weight": np.arange(15).reshape(3, 5).astype(float),
            "blk.0.ffn_down.weight": np.arange(12).reshape(3, 4).astype(float),
            "blk.1.attn_output.weight": np.arange(6).reshape(3, 2).astype(float),
            "blk.1.ffn_down.weight": np.arange(12, 24).reshape(3, 4).astype(float),
        }

        def weight_shape(self, name):
            rows, columns = self.weights[name].shape
            return columns, rows, 1, 1

        def weight_rows(self, name, start, count, width):
            assert self.weights[name].shape[1] == width
            return self.weights[name][start:start+count].astype(np.float32)

    bridge = FakeBridge()
    before = {name: w.copy() for name, w in bridge.weights.items()}
    u = np.array([0, .6, .8], np.float32)
    out = tmp_path / "synthetic.npz"
    provenance = {"model_key": bridge.model_key, "direction_source": "synthetic test"}
    manifest = export_factors(bridge, u, out, provenance)
    assert manifest["native_adapter"] == "synapse-callback-rank-one-v1"
    assert manifest["llama_cpp_lora"] is False
    assert manifest["readout"] == "unchanged"
    with np.load(out, allow_pickle=False) as data:
        assert json.loads(str(data["manifest"])) == manifest
        assert set(data.files) == set(bridge.weights) | {"B", "manifest"}
        for name, w in bridge.weights.items():
            h = np.arange(w.shape[1]) / 3
            for strength in (0, .25, 1, -.25):
                reference = (np.eye(3) - strength * np.outer(u, u)) @ w @ h
                factored = w @ h + strength * data["B"] * (data[name] @ h)
                np.testing.assert_allclose(factored, reference, atol=1e-5)
            np.testing.assert_array_equal(w, before[name])
    with pytest.raises(ValueError, match="new"):
        export_factors(bridge, u, out, provenance)
    with pytest.raises(ValueError, match="provenance"):
        export_factors(bridge, u, tmp_path / "wrong.npz", {"model_key": "other"})
    import residual_projection
    monkeypatch.setattr(residual_projection, "artifact_path", lambda value: Path(value))
    loaded_u, bank, receipt = load_factors(bridge, str(out))
    np.testing.assert_array_equal(loaded_u, u)
    expected_bank = np.concatenate([u.astype(float) @ w for w in bridge.weights.values()])
    np.testing.assert_allclose(bank, expected_bank, atol=2e-6)
    assert receipt["manifest"] == manifest
    bridge.model_key = "different"
    with pytest.raises(ValueError, match="checkpoint"):
        load_factors(bridge, str(out))


def test_factor_loader_rejects_oversized_npy_header_without_allocating(tmp_path, monkeypatch):
    import io
    import zipfile
    import residual_projection
    path = tmp_path / "malformed.npz"
    payload = io.BytesIO()
    np.lib.format.write_array_header_1_0(payload, {
        "shape": (10**12,), "fortran_order": False, "descr": "<f4",
    })
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("B.npy", payload.getvalue())
    monkeypatch.setattr(residual_projection, "artifact_path", lambda value: Path(value))
    with pytest.raises(ValueError, match="header"):
        load_factors(None, str(path))


def test_artifact_namespace_rejects_outside_files():
    from residual_projection import artifact_path
    for value in ["/etc/passwd", ".synapse/experiments/../../../outside.npz", "", 1]:
        with pytest.raises(ValueError):
            artifact_path(value)


def test_bad_artifact_rejected_before_inference_and_temporary_inspector_closed(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    import gguf_runtime
    import residual_projection
    runtime = object.__new__(gguf_runtime.GGUFRuntime)
    runtime.model, runtime.inspector = None, None
    runtime.lock = threading.RLock()
    closed = []

    def inspect():
        runtime.inspector = SimpleNamespace(close=lambda: closed.append(True))
        return runtime.inspector

    def forbidden():
        raise AssertionError("Invalid artifact must not load inference")

    def reject(*args):
        raise ValueError("Malformed archive fixture")

    path = tmp_path / "invalid.npz"
    path.write_bytes(b"bad archive")
    monkeypatch.setattr(residual_projection, "artifact_path", lambda value: path)
    monkeypatch.setattr(residual_projection, "load_factors", reject)
    runtime.inspection_bridge = inspect
    runtime.ensure_loaded = forbidden
    with pytest.raises(ValueError, match="Malformed"):
        runtime.residual_probe({"model_key": gguf_runtime.MODEL_KEY, "text": "Hello",
                                "layer": 0, "projection": {"artifact": "fixture.npz", "strength": .25}})
    assert closed == [True] and runtime.inspector is None


def test_http_residual_boundary_without_model(monkeypatch):
    from fastapi.testclient import TestClient
    import server
    from types import SimpleNamespace
    calls = []

    def probe(body):
        calls.append(body)
        projection_request(body, "fixture", 3, 2)
        return {"fixture_only": True, "saved_weights_modified": False}

    monkeypatch.setattr(server, "GGUF", True)
    monkeypatch.setattr(server, "NO_MODEL", False)
    monkeypatch.setattr(server, "rt", SimpleNamespace(residual_probe=probe))
    with TestClient(server.app, base_url="http://127.0.0.1:8791") as client:
        request = {"model_key": "fixture", "text": "Hello", "layer": 1, "max_new": 0}
        good = client.post("/api/agent/residual", json=request)
        assert good.status_code == 200 and good.json()["fixture_only"] is True
        assert calls == [request]
        assert client.post("/api/agent/residual", json=dict(request, model_key="wrong")).status_code == 422
        assert client.post("/api/agent/residual", json=[]).status_code == 422
        assert client.post("/api/agent/residual", content="not json").status_code == 422
        assert client.post("/api/agent/residual", content=" " * (256 * 1024 + 1)).status_code == 422
        assert client.post("/api/agent/residual", json=request,
                           headers={"origin": "https://example.com"}).status_code == 403
