#!/usr/bin/env python3
"""Local Qwen lifecycle, guarded by measured available RAM and a cgroup limit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/"models/Qwen3.5-4B-Q4_K_M.gguf"
UNIT = "qwen-openweights-lab.service"
BASE = "http://127.0.0.1:8087"
MIN_AVAILABLE = 5 * 1024**3

def available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])*1024
    raise RuntimeError("Cannot read MemAvailable")

def binary(name):
    matches = [p for p in (ROOT/"runtime").rglob(name) if p.is_file() and os.access(p, os.X_OK)]
    if len(matches) != 1:
        raise SystemExit(f"Expected one installed {name}; found {len(matches)}. Complete install_assets.py first.")
    return str(matches[0])

def active():
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", UNIT]).returncode == 0

def get(path, timeout=3):
    with urllib.request.urlopen(BASE+path, timeout=timeout) as r:
        return json.load(r)

def start():
    if active():
        print("Model service is already running.")
        return
    if not MODEL.is_file():
        raise SystemExit("The model download is not complete.")
    lease=ROOT/".synapse/local-model-cpu.lock"
    lease.parent.mkdir(exist_ok=True)
    with open(lease,"a") as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Synapse or another Qwen probe has the model loaded. Run synapse stop before qwen-lab chat.") from None
    ram = available()
    if ram < MIN_AVAILABLE:
        raise SystemExit(f"Model remains stopped: {ram/1024**3:.2f} GiB RAM available; need at least 5.00 GiB. Close unused apps, then retry.")
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", 8087))
        except OSError as e:
            raise SystemExit(f"Local port 8087 is in use: {e}")
    exe = binary("llama-server")
    subprocess.run(["systemctl", "--user", "reset-failed", UNIT], capture_output=True)
    command = [
        "systemd-run", "--user", "--collect", "--unit="+UNIT,
        "--property=MemoryMax=4G", "--property=MemorySwapMax=0",
        "--property=CPUQuota=300%", "--property=Nice=10", "--property=TasksMax=96",
        "--working-directory="+str(ROOT),
        "flock", "--nonblock", "--no-fork", str(lease), exe, "--model", str(MODEL), "--alias", "qwen3.5-4b-q4km",
        "--host", "127.0.0.1", "--port", "8087", "--n-gpu-layers", "0",
        "--ctx-size", "16384", "--parallel", "1", "--threads", "4", "--threads-batch", "4",
        "--threads-http", "2", "--batch-size", "128", "--ubatch-size", "64",
        "--jinja", "--chat-template-kwargs", '{"enable_thinking":false}', "--reasoning", "off",
    ]
    subprocess.run(command, check=True)
    deadline = time.monotonic()+60
    while time.monotonic() < deadline:
        try:
            if get("/health").get("status") == "ok":
                print("Qwen is ready at "+BASE)
                return
        except (OSError, ValueError):
            pass
        if not active():
            raise SystemExit("Model service exited. Inspect: journalctl --user -u "+UNIT+" -n 60")
        time.sleep(1)
    raise SystemExit("Service started but is not ready after 60 seconds. Inspect qwen-lab status and the service journal.")

def smoke():
    start()
    request = urllib.request.Request(BASE+"/v1/chat/completions", data=json.dumps({
        "model": "qwen3.5-4b-q4km", "messages": [{"role":"user", "content":"What is 17 plus 25? Reply with the number only."}],
        "temperature": 0, "seed": 42, "max_tokens": 32,
    }).encode(), headers={"Content-Type":"application/json"})
    before = time.monotonic()
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    receipt = {"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()), "elapsed_seconds":time.monotonic()-before,
               "purpose":"installation smoke test, not a performance benchmark", "response":result}
    out = ROOT/"evidence"/("inference-smoke-"+time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())+".json")
    out.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))
    if result["choices"][0]["message"].get("content", "").strip() != "42":
        raise SystemExit("API replied, but exact-answer smoke check did not pass. Raw response was preserved.")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status","start","stop","chat","smoke"])
    args, extra=parser.parse_known_args()
    if extra and args.command != "chat":
        parser.error("Extra arguments are supported only after chat")
    if args.command == "status":
        report={"workspace":str(ROOT),"model_present":MODEL.is_file(),"available_ram_gib":round(available()/1024**3,2),
                "required_available_ram_gib":5,"service_active":active(),"free_disk_gib":round(shutil.disk_usage(ROOT).free/1024**3,2)}
        if report["service_active"]:
            try:report["health"]=get("/health")
            except OSError as e:report["health_error"]=str(e)
        print(json.dumps(report,indent=2))
    elif args.command == "stop":
        if active():subprocess.run(["systemctl","--user","stop",UNIT],check=True)
        print("Model stopped.")
    elif args.command == "start":start()
    elif args.command == "smoke":smoke()
    elif args.command == "chat":
        start()
        os.chdir(ROOT/"sandbox")
        os.environ["OPENCODE_CONFIG"]=str(ROOT/"sandbox/opencode.json")
        os.execv(binary("opencode"),["opencode",*extra])

if __name__ == "__main__":
    main()
