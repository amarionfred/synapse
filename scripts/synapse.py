#!/usr/bin/env python3
"""Small local launcher. Model selection is process-scoped; checkpoints never share caches."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ROOT / ".synapse/settings.json"
UNIT = "synapse-explorer.service"
SMALL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def main():
    p = argparse.ArgumentParser(description="Synapse neuron explorer and Codex workspace")
    p.add_argument("action", choices=["start", "serve", "stop", "status", "codex", "index"])
    p.add_argument("--model", help="Hugging Face ID or local Transformers checkpoint")
    p.add_argument("--revision", default=None)
    p.add_argument("--author-map", action="store_true", help="Lyra's cached Qwen map; no model loaded")
    p.add_argument("--port", type=int, default=8791)
    args, extra = p.parse_known_args()
    if args.action == "stop":
        return subprocess.call(["systemctl", "--user", "stop", UNIT])
    if args.action == "status":
        import urllib.request
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/api/agent/overview", timeout=5) as r:
                print(json.dumps(json.load(r), indent=2))
            return 0
        except OSError as e:
            print(f"Synapse is stopped or unreachable: {e}")
            return 1
    if args.action == "codex":
        sys.path.insert(0, str(ROOT / "explorer"))
        from codex_agent import command
        # Interactive CLI uses the same workspace and MCP as the web panel.
        cmd = command(ROOT, args.port, "terminal", args.model)
        cmd = cmd[:cmd.index("exec")] + extra
        os.execv(cmd[0], cmd)
    settings = json.loads(SETTINGS.read_text()) if SETTINGS.exists() else {}
    model = args.model or settings.get("model", SMALL)
    revision = args.revision or (settings.get("revision", "main") if model == settings.get("model") else "main")
    bundled = args.author_map or (not args.model and settings.get("author_map", False))
    if bundled:
        model, revision = "Qwen/Qwen3.5-4B", "main"
    env = dict(os.environ, EXPLORER_MODEL=model, EXPLORER_REVISION=revision,
               EXPLORER_NO_MODEL=str(int(bundled)), EXPLORER_BUNDLED_CACHE=str(int(bundled)), PORT=str(args.port),
               HF_HOME=str(ROOT / ".synapse/huggingface"), TOKENIZERS_PARALLELISM="false")
    py = ROOT / ".venv/bin/python"
    if not py.exists():
        p.error("Install dependencies first: uv sync")
    SETTINGS.parent.mkdir(exist_ok=True)
    SETTINGS.write_text(json.dumps({"model": model, "revision": revision, "author_map": bundled}, indent=2)+"\n")
    if args.action == "index":
        os.execve(str(py), [str(py), str(ROOT/"scripts/build_indexes.py"), *extra], env)
    if args.action == "serve":
        os.execve(str(py), [str(py), str(ROOT / "explorer/server.py")], env)
    active = subprocess.run(["systemctl", "--user", "is-active", "--quiet", UNIT]).returncode == 0
    if active:
        print("Synapse is already running. Stop it before selecting another checkpoint: synapse stop")
        return 1
    subprocess.run(["systemctl", "--user", "reset-failed", UNIT], capture_output=True)
    cmd = ["systemd-run", "--user", "--collect", "--unit="+UNIT,
           "--property=MemoryMax=3G", "--property=MemorySwapMax=0", "--property=CPUQuota=300%",
           "--property=Nice=10", "--working-directory="+str(ROOT)]
    for key in ("EXPLORER_MODEL", "EXPLORER_REVISION", "EXPLORER_NO_MODEL", "EXPLORER_BUNDLED_CACHE", "PORT", "HF_HOME", "TOKENIZERS_PARALLELISM"):
        cmd += [f"--setenv={key}={env[key]}"]
    code = subprocess.call(cmd + [str(py), str(ROOT/"explorer/server.py")])
    if not code:
        print(f"Synapse starting at http://127.0.0.1:{args.port} · {model}")
        print("The model loads when requested. synapse status checks readiness.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
