import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"explorer"))
import pytest
from codex_agent import translate, valid_chat, command


def test_event_translation():
    call = {"id": "t1", "type": "mcp_tool_call", "tool": "overview", "arguments": {}}
    assert translate({"type": "item.started", "item": call})[0]["name"] == "overview"
    call.update(result={"content": [{"type":"text", "text":"model data"}]}, status="completed")
    assert "model data" in translate({"type":"item.completed", "item":call})[0]["output"]
    assert translate({"type":"turn.failed", "error":{"message":"disconnected"}})[0]["error"] == "disconnected"
    assert translate({"type":"item.completed", "item":{"type":"agent_message","text":"hello"}})[0]["text"] == "hello\n"


def test_chat_path_validation():
    with pytest.raises(ValueError):
        valid_chat("../../auth")
    assert valid_chat(valid_chat(None))


def test_cli_arguments_preserve_prompt_boundary(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/codex")
    (tmp_path/"config.toml").write_text('[mcp_servers.other-server]\ncommand="python"\n')
    cmd = command(tmp_path, 8791, valid_chat(None), "model-name", "thread-id")
    assert cmd[-5:] == ["exec", "resume", "thread-id", "--json", "-"]
    assert "workspace-write" in cmd
    assert "mcp_servers.synapse.required=true" in cmd
    assert 'mcp_servers.synapse.default_tools_approval_mode="approve"' in cmd
    assert "mcp_servers.other-server.enabled=false" in cmd
