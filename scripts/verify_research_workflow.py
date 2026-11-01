#!/usr/bin/env python3
"""Serial offline verification; no trained checkpoint load or downloads.

On Linux/systemd this re-executes under 1 GiB RAM, no swap and one CPU. Optional
--ui checks real client functions with synthetic DOM/HTTP fixtures; --native
checks actual GGML CPU callbacks using already installed pinned headers/runtime.
Neither is live-checkpoint or real-browser numerical validation.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ui", action="store_true", help="Also run the Node UI contract checks")
    parser.add_argument("--native", action="store_true", help="Also compile/run native CPU callback fixtures")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update(EXPLORER_NO_MODEL="1", EXPLORER_BUNDLED_CACHE="1", EXPLORER_PRELOAD="0",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if sys.platform == "linux" and shutil.which("systemd-run") and "XDG_RUNTIME_DIR" in env and not env.get("SYNAPSE_VERIFY_BOUNDED"):
        env["SYNAPSE_VERIFY_BOUNDED"] = "1"
        return subprocess.call(["systemd-run", "--user", "--scope", "--quiet",
            "-p", "MemoryMax=1G", "-p", "MemorySwapMax=0", "-p", "CPUQuota=100%",
            sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], cwd=root, env=env)
    def run(command):
        return subprocess.call(command, cwd=root, env=env)
    print("Offline implementation checks; no live checkpoint inference.", flush=True)
    code = run([sys.executable, "-m", "pytest", "-q", "tests"])
    if not code and args.ui:
        for fixture in ("tests/map_selection_ui.cjs", "tests/research_ui.cjs"):
            code = run(["node", fixture])
            if code:
                break
    if not code and args.native:
        runtime, headers = root / "runtime/llama/llama-b11669", root / ".synapse/llama-headers"
        if not (runtime / "libllama.so").exists() or not (headers / "llama.h").exists():
            parser.error("Native fixtures require the existing pinned runtime/headers; this check does not download them")
        for name in ("projection", "scoped"):
            binary = root / ".synapse" / ("test-native-" + name)
            code = run(["c++", "-std=c++17", "-O1", str(root / "tests" / ("native_" + name + ".cpp")),
                        "-I" + str(headers), "-L" + str(runtime), "-Wl,-rpath," + str(runtime),
                        "-lllama", "-lggml", "-lggml-base", "-o", str(binary)])
            if not code:
                code = run([str(binary), str(runtime)])
            if code:
                break
    if sys.platform == "linux":
        group = next((line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                      if line.startswith("0::")), None)
        if group:
            cgroup = Path("/sys/fs/cgroup") / group.lstrip("/")
            print(json.dumps({"memory_peak_bytes": (cgroup / "memory.peak").read_text().strip(),
                              "memory_events": (cgroup / "memory.events").read_text().strip()}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
