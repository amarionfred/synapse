#!/usr/bin/env python3
"""Build the small native bridge against installed llama.cpp b11669, not a new runtime."""

import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "620d2225c"


def main():
    runtime = ROOT / "runtime/llama/llama-b11669"
    if not (runtime / "llama-server").exists():
        raise SystemExit(
            "Install the pinned llama.cpp runtime first: python scripts/install_assets.py"
        )
    version = subprocess.check_output(
        [str(runtime / "llama-server"), "--version"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    if "build 11669, commit 620d2225c" not in version:
        raise SystemExit(
            "Installed llama.cpp ABI differs from the bridge's pinned headers"
        )
    headers = ROOT / ".synapse/llama-headers"
    headers.mkdir(parents=True, exist_ok=True)
    for path, sha in json.loads(
        (ROOT / "native/llama-bridge/headers.json").read_text()
    ).items():
        dest = headers / Path(path).name
        data = dest.read_bytes() if dest.exists() else b""
        if hashlib.sha256(data).hexdigest() != sha:
            with urllib.request.urlopen(
                f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{COMMIT}/{path}",
                timeout=30,
            ) as r:
                data = r.read()
            if hashlib.sha256(data).hexdigest() != sha:
                raise SystemExit(f"Header hash mismatch: {path}")
            dest.write_bytes(data)
    output = ROOT / ".synapse/libsynapse-llama.so"
    tmp = output.with_suffix(".so.build")
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-shared",
            str(ROOT / "native/llama-bridge/bridge.cpp"),
            "-I" + str(headers),
            "-L" + str(runtime),
            "-Wl,-rpath," + str(runtime),
            "-lllama",
            "-lggml",
            "-lggml-base",
            "-o",
            str(tmp),
        ],
        check=True,
    )
    tmp.replace(output)
    print(f"Built {output} against llama.cpp b11669 ({COMMIT})")


if __name__ == "__main__":
    main()
