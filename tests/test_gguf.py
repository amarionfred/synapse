"""Format dispatch and checkpoint-selection regressions; live native checks are explicit."""

import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))
from gguf_metadata import describe, identity, metadata


def write_gguf(path, arch="qwen35"):
    fields = {
        "general.architecture": arch,
        "qwen35.block_count": 32,
        "qwen35.embedding_length": 2560,
        "qwen35.feed_forward_length": 9216,
        "qwen35.attention.head_count": 16,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.full_attention_interval": 4,
        "tokenizer.chat_template": "{{ messages[0].content }}",
    }
    string = lambda x: struct.pack("<Q", len(x.encode())) + x.encode()
    data = b"GGUF" + struct.pack("<IQQ", 3, 0, len(fields))
    for k, v in fields.items():
        data += string(k) + struct.pack("<I", 8 if isinstance(v, str) else 4)
        data += string(v) if isinstance(v, str) else struct.pack("<I", v)
    path.write_bytes(data)


def test_gguf_identity_isolates_files_and_edits(tmp_path):
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    write_gguf(a)
    write_gguf(b)
    assert identity(a)[1] != identity(b)[1]
    old = identity(a)[1]
    a.write_bytes(a.read_bytes() + b"edit")
    assert identity(a)[1] != old
    arch = describe(a)
    assert (arch["n_layers"], arch["inter"]) == (32, 9216)
    assert arch["layer_types"][:4] == ["linear_attention"] * 3 + ["full_attention"]
    assert metadata(a)["tokenizer.chat_template"] == "{{ messages[0].content }}"


def test_gguf_unknown_arch_and_truncated_files_fail(tmp_path):
    p = tmp_path / "model.gguf"
    write_gguf(p, "qwen35moe")
    with pytest.raises(ValueError, match="no validated"):
        describe(p)
    for data in (
        b"",
        b"GGUF",
        b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 2**40),
    ):
        p.write_bytes(data)
        with pytest.raises(ValueError):
            metadata(p)


def test_rejected_start_keeps_selected_model(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/synapse.py"
    spec = importlib.util.spec_from_file_location("synapse_launcher_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = tmp_path / "settings.json"
    original = json.dumps({"model": "active-model", "revision": "pinned"})
    settings.write_text(original)
    monkeypatch.setattr(module, "SETTINGS", settings)
    monkeypatch.setattr(
        module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(sys, "argv", [str(path), "start", "--model", "another-model"])
    assert module.main() == 1
    assert settings.read_text() == original
