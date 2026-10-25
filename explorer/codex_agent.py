"""Codex CLI JSONL -> explorer events. Authentication stays in the user's CLI."""
import asyncio
from collections import deque
import json
import os
from pathlib import Path
import shutil
import sys
import tomllib
import uuid

EXPLORER_TOOLS = ["overview", "neuron", "rollouts", "cone", "web_summary", "search_lens", "top_units", "set_name",
                  "names", "steer", "activations", "compare", "logprobs", "top_firing", "select_neuron",
                  "highlight_neurons", "show_text", "show_search", "camera", "annotate_neurons", "open_tab"]


def command(root, port, chat, model=None, thread=None):
    cli = shutil.which("codex")
    if not cli:
        raise RuntimeError("Codex CLI is not installed; install it and run codex login first.")
    args = [cli, "-C", str(root), "-a", "never", "-s", "workspace-write"]
    def config(key, value):
        args.extend(["-c", f"{key}={json.dumps(value)}"])
    # Preserve the user's model/provider/auth configuration, without unrelated MCP servers.
    cfg_path = Path(os.getenv("CODEX_HOME", str(Path.home()/".codex"))) / "config.toml"
    if cfg_path.exists():
        with cfg_path.open("rb") as f:
            cfg = tomllib.load(f)
        for name in cfg.get("mcp_servers", {}):
            # Codex's dotted override parser treats quote characters literally.
            # MCP server names in the CLI are simple identifiers (hyphens allowed).
            if "." in name:
                raise RuntimeError(f"MCP server name {name!r} contains a dot; use a dedicated Codex profile for this installation")
            config(f'mcp_servers.{name}.enabled', False)
    config("sandbox_workspace_write.network_access", True)
    config("mcp_servers.synapse.command", sys.executable)
    config("mcp_servers.synapse.args", [str(Path(root)/"explorer/agent_mcp.py")])
    config("mcp_servers.synapse.enabled", True)
    config("mcp_servers.synapse.required", True)
    # The person explicitly enabled this local explorer. Grant only its known
    # research/UI tools; this does not change shell or other-server approvals.
    config("mcp_servers.synapse.enabled_tools", EXPLORER_TOOLS)
    config("mcp_servers.synapse.default_tools_approval_mode", "approve")
    config("mcp_servers.synapse.env.EXPLORER_URL", f"http://127.0.0.1:{port}")
    config("mcp_servers.synapse.env.EXPLORER_CHAT", chat)
    config("mcp_servers.synapse.tool_timeout_sec", 900)
    if model:
        args += ["-m", model]
    args += ["exec"]
    if thread:
        args += ["resume", thread]
    return args + ["--json", "-"]


def translate(event):
    kind, item = event.get("type"), event.get("item") or {}
    typ, ident = item.get("type"), item.get("id")
    if kind == "item.completed" and typ == "agent_message":
        return [{"t": "text", "text": item.get("text", "") + "\n"}]
    if kind == "item.started" and typ in {"mcp_tool_call", "command_execution", "file_change", "web_search"}:
        return [{"t": "tool", "id": ident, "name": item.get("tool") or typ,
                 "input": item.get("arguments") or {k: item[k] for k in ("command", "changes", "query") if k in item}}]
    if kind == "item.completed" and typ in {"mcp_tool_call", "command_execution", "file_change", "web_search"}:
        output = item.get("result") or item.get("aggregated_output") or item.get("error") or item.get("changes") or "completed"
        output = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
        return [{"t": "tool_result", "id": ident, "error": item.get("status") == "failed" or bool(item.get("error")),
                 "output": output[:6000]}]
    if kind in {"error", "turn.failed"}:
        error = event.get("message") or event.get("error") or "Codex turn failed"
        return [{"t": "error", "error": error if isinstance(error, str) else error.get("message", str(error))}]
    if kind == "turn.completed":
        return [{"t": "done", "usage": event.get("usage")}]
    return []


def valid_chat(value):
    return str(uuid.UUID(value)) if value else str(uuid.uuid4())


async def stream(body, root, port, cid, chat, prompt, stop, attachments, checkpoint):
    directory = Path(root)/".synapse/agents"
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory/f"{cid}.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state and state.get("checkpoint", checkpoint) != checkpoint:
        yield json.dumps({"t": "error", "error": "Start a new conversation for this checkpoint."}) + "\n"
        return
    model = body.get("agent_model") or os.getenv("EXPLORER_CODEX_MODEL")
    if state and state.get("model") != model:
        yield json.dumps({"t": "error", "error": "Start a new conversation when changing the agent model."}) + "\n"
        return
    await stop(chat.get("proc"))
    msg = body["message"]
    if body.get("layer") is not None:
        msg = f"The selected neuron is L{int(body['layer'])} N{int(body['n'])}.\n\n{msg}"
    if attachments:
        msg += "\n\nAttached local files (read only as relevant):\n" + "\n".join(attachments)
    if not state.get("thread"):
        msg = prompt + "\n\nPerson's request:\n" + msg
    q = chat["queue"]
    proc, tasks = None, []
    try:
        proc = chat["proc"] = await asyncio.create_subprocess_exec(
            *command(root, port, cid, model, state.get("thread")), cwd=root,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True, limit=1 << 24)
        async def stdout():
            async for line in proc.stdout:
                q.put_nowait(("codex", line))
            q.put_nowait(("end", None))
        errors = deque(maxlen=30)
        async def stderr():
            async for line in proc.stderr:
                errors.append(line.decode(errors="replace"))
        tasks = [asyncio.create_task(stdout()), asyncio.create_task(stderr())]
        proc.stdin.write(msg.encode())
        await proc.stdin.drain()
        proc.stdin.close()
        yield json.dumps({"t": "chat", "chat": cid, "provider": "codex"}) + "\n"
        while True:
            kind, data = await q.get()
            if kind == "ui":
                yield json.dumps({"t": "ui", **data}) + "\n"
            elif kind == "end":
                await proc.wait()
                await tasks[1]
                if proc.returncode:
                    yield json.dumps({"t": "error", "error": "".join(errors)[-1500:] or f"Codex exited {proc.returncode}"}) + "\n"
                break
            elif kind == "codex":
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                if event.get("type") == "thread.started":
                    state.update(thread=event["thread_id"], model=model, provider="codex", checkpoint=checkpoint)
                    tmp = state_path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(state) + "\n")
                    tmp.replace(state_path)
                for out in translate(event):
                    yield json.dumps(out) + "\n"
    except (OSError, RuntimeError) as e:
        yield json.dumps({"t": "error", "error": str(e)}) + "\n"
    finally:
        for task in tasks:
            task.cancel()
        await stop(proc)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
