"""Live connection endpoints for the installed Qwen; run with the project Python."""

import json
import time
import urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8791"
result = {
    "purpose": "connection-view integration on the installed quantized Qwen weights"
}
receipt = ROOT / "evidence/gguf-connections-api.json"


def get(path, binary=False):
    start = time.monotonic()
    with urllib.request.urlopen(BASE + path, timeout=180) as r:
        raw, headers = r.read(), dict(r.headers)
    return (raw if binary else json.loads(raw)), headers, time.monotonic() - start


for name, path, binary in [
    ("connections", "/api/connections?layer=14&n=8949&k=24", False),
    ("cached", "/api/connections?layer=14&n=8949&k=24", False),
    ("web", "/api/web?layer=14&n=8949&first=5&hops=1", True),
    ("two_hops", "/api/web?layer=14&n=8949&first=2&k=1&hops=2&zmin=0", True),
    ("summary", "/api/web/summary?layer=14&n=8949&first=5&hops=1", False),
    ("cone", "/api/agent/cone?layer=14&n=8949&k=1&hops=2", False),
]:
    data, headers, seconds = get(path, binary)
    if binary:
        data = np.frombuffer(data, np.int32).reshape(-1, 6).tolist()
        assert data and all(
            0 <= r[0] < r[2] < 32 and 0 <= r[1] < 9216 and 0 <= r[3] < 9216
            for r in data
        )
        if name == "web":
            assert len(data) == 10
        if name == "two_hops":
            assert any(abs(r[5]) == 2 for r in data)
    if name in ("connections", "cached"):
        assert data["available"] and len(data["up"]) == 24 and len(data["down"]) == 24
        assert all(c["layer"] < 14 and c["lens"] for c in data["up"])
        assert all(c["layer"] > 14 and c["lens"] for c in data["down"])
        if name == "cached":
            assert data["cached"]
    if name == "summary":
        assert (
            data["src"] == "on-demand logit"
            and data["links"] == 10
            and data["down"]["toward"]
        )
    if name == "cone":
        assert data["nodes"] and data["mode"] == "on-demand"
    result[name] = {
        "seconds": seconds,
        "response": data,
        "headers": {k: v for k, v in headers.items() if k.lower().startswith("x-web")},
    }
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(name, "passed", round(seconds, 3), flush=True)
result["passed"] = True
receipt.write_text(json.dumps(result, indent=2) + "\n")
