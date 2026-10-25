#!/usr/bin/env python3
"""Install pinned public release assets; verify upstream SHA256/SHA512 before use."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ASSETS = [
    dict(name="Qwen3.5-4B-Q4_K_M.gguf", directory="models",
         url="https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf",
         sha256="00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4", size=2740937888),
    dict(name="llama-b11669-bin-ubuntu-x64.tar.gz", directory="downloads",
         url="https://github.com/ggml-org/llama.cpp/releases/download/b11669/llama-b11669-bin-ubuntu-x64.tar.gz",
         sha256="cf07c1c112d709bd67e522017471de57ed0ec8bea031d732c5422351c07f65ac", size=17851611),
    dict(name="opencode-linux-x64-1.19.4.tgz", directory="downloads",
         url="https://registry.npmjs.org/opencode-linux-x64/-/opencode-linux-x64-1.19.4.tgz",
         sha512="ad0fe5de3989a317fc6589c16ab820f5a954465cfa6ee122972b06383b7391c99c62d719f752ce805d37e285141c7425f3fcb585b4530ef83aba3a948d0b0c5f",
         size=79146075),
]

def digest(path, algorithm):
    with path.open("rb") as f:
        return hashlib.file_digest(f, algorithm).hexdigest()

def download(asset):
    target = ROOT/asset["directory"]/asset["name"]
    target.parent.mkdir(exist_ok=True)
    algorithm = "sha512" if "sha512" in asset else "sha256"
    if not target.exists():
        part = target.with_suffix(target.suffix+".part")
        print("Downloading", asset["name"], flush=True)
        subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "3",
                        "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120",
                        "--continue-at", "-", "--output", str(part), asset["url"]], check=True)
        if (asset["size"] is not None and part.stat().st_size != asset["size"]) or digest(part, algorithm) != asset[algorithm]:
            raise RuntimeError("Size/digest mismatch: "+str(part))
        part.rename(target)
    elif (asset["size"] is not None and target.stat().st_size != asset["size"]) or digest(target, algorithm) != asset[algorithm]:
        raise RuntimeError("Existing asset differs from pinned upstream: "+str(target))
    print("Verified", asset["name"], flush=True)
    return target

def main():
    missing = sum(max(0, (a["size"] or a["estimated_size"]) -
                      ((ROOT/a["directory"]/(a["name"]+".part")).stat().st_size
                       if (ROOT/a["directory"]/(a["name"]+".part")).exists() else 0))
                  for a in ASSETS if not (ROOT/a["directory"]/a["name"]).exists())
    if shutil.disk_usage(ROOT).free < missing + 1100 * 1024**2:
        raise SystemExit("Insufficient disk space: downloads plus extraction and 750 MiB reserve required")
    with ThreadPoolExecutor(max_workers=3) as pool:
        targets = list(pool.map(download, ASSETS))
    for target in targets:
        if target.name.endswith((".tar.gz", ".tgz")):
            dest = ROOT/"runtime"/("llama" if target.name.startswith("llama-") else "opencode")
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(target) as archive:
                archive.extractall(dest, filter="data")
    (ROOT/"evidence"/"installed-assets.json").write_text(json.dumps(ASSETS, indent=2)+"\n")
    print("All pinned assets installed and verified.", flush=True)

if __name__ == "__main__":
    main()
