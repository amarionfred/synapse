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


def test_mcp_protocol_exposes_allowed_research_tools_and_preserves_http_custody(tmp_path):
    import json
    import os
    import subprocess
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from codex_agent import EXPLORER_TOOLS
    observed = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            observed.append(("GET", self.path, None))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"fixture":true}')
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            observed.append(("POST", self.path, body))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"fixture":true}')
        def log_message(self, *args):
            pass
    http = HTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    calls = [
        ("score_sequence", {"text": " p\n", "completion": " 12"}),
        ("capture_selected", {"text": "p", "selection": [[0, 0, 3]]}),
        ("research_job", {"action": "submit", "spec": {"split": "development"}}),
        ("research_job", {"action": "feedback", "id": "job-fixture"}),
        ("intervention_preset", {"action": "restore", "payload": {"id": None}}),
        ("local_researcher", {"action": "choose", "id": "episode-fixture", "feedback": []}),
    ]
    messages = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
    messages += [{"jsonrpc": "2.0", "id": i+3, "method": "tools/call",
                  "params": {"name": name, "arguments": arguments}} for i, (name, arguments) in enumerate(calls)]
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "explorer/agent_mcp.py")],
                                input="\n".join(json.dumps(m) for m in messages)+"\n", capture_output=True,
                                text=True, timeout=15, check=True,
                                env={**os.environ, "EXPLORER_URL": f'http://127.0.0.1:{http.server_port}'})
    finally:
        http.shutdown()
        worker.join(2)
        http.server_close()
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == len(messages)
    tools = rows[1]["result"]["tools"]
    names = [tool["name"] for tool in tools]
    assert len(names) == len(set(names))
    assert set(EXPLORER_TOOLS) == set(names)
    assert all(not row["result"].get("isError") for row in rows[2:])
    assert observed == [
        ("POST", "/api/research/score", calls[0][1]),
        ("POST", "/api/research/capture", calls[1][1]),
        ("POST", "/api/research/jobs", {"split": "development"}),
        ("GET", "/api/research/jobs/job-fixture/feedback", None),
        ("POST", "/api/research/presets/restore", {"id": None}),
        ("POST", "/api/research/researchers/episode-fixture/choose", {"feedback": [], "timeout_seconds": 60}),
    ]
