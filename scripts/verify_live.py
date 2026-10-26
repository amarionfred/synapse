"""Bounded live integration checks. This measures plumbing, not model intelligence."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8791")
p.add_argument("--out", default="evidence/synapse-live-smoke.json")
a = p.parse_args()
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
results = {"purpose": "integration smoke; not a capability or performance benchmark"}


def request(path, body=None):
    req = urllib.request.Request(a.url+path, None if body is None else json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def record():
    out.write_text(json.dumps(results, indent=2)+"\n")


try:
    # systemd starts before Python has finished importing the server.
    deadline=time.monotonic()+30
    while True:
        try:
            request("/api/status")
            break
        except OSError:
            if time.monotonic()>=deadline:
                raise
            time.sleep(0.5)
    results["overview"] = info = request("/api/model/load", {})
    l, n = min(10, info["arch"]["n_layers"]-1), min(3, info["arch"]["inter"]-1)
    for name, path, body in [
        ("read", "/api/reads", {"layer":0,"n":0,"text":"The capital of France is Paris.","chat":False}),
        ("logprobs", "/api/agent/logprobs", {"text":"The capital of France is","chat":False,
                                           "neurons":[{"layer":l,"n":n,"dose":0.05}],"targets":[" Paris"],"k":4}),
        ("steer_zero", "/api/agent/steer", {"layer":l,"n":n,"prompt":"Name a color.","dose":0,"max_new":8}),
        ("steer_signed", "/api/agent/steer", {"layer":l,"n":n,"prompt":"Name a color.","dose":0.05,"max_new":8}),
        ("infer", "/api/infer", {"text":"The river is quiet.","chat":False})]:
        started = time.monotonic()
        results[name] = {"request":body, "response":request(path,body), "elapsed_seconds":time.monotonic()-started}
        record()
        print(name, "ok", round(time.monotonic()-started, 2), flush=True)
    z = results["steer_zero"]["response"]
    assert z["up"] == z["baseline"] == z["down"], "Zero-dose continuations differ"
    results["zero_dose_equal"] = True
    print("zero-dose continuations identical")
except Exception as e:
    results["error"] = repr(e)
    raise
finally:
    record()
