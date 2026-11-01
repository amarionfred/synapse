#!/usr/bin/env python3
"""Validate or operate bounded research jobs on the existing Synapse server.

This client never constructs a Runtime or loads a model. A submitted spec is
sent to the existing serial evaluator; --wait interrupts request cancellation
on Ctrl-C and reports its current acknowledgment state.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explorer"))
from experiment_schema import canonical, digest, experiment_spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8791)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate/freeze a spec without contacting inference")
    validate.add_argument("spec", type=Path)
    submit = commands.add_parser("submit", help="Submit one frozen bounded evaluator job")
    submit.add_argument("spec", type=Path)
    submit.add_argument("--wait", type=float, default=0, metavar="SECONDS")
    for name in ("status", "cancel", "resume"):
        command = commands.add_parser(name)
        command.add_argument("job_id")
        if name == "resume":
            command.add_argument("--wait", type=float, default=0, metavar="SECONDS")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    waiting = getattr(args, "wait", 0)
    if not math.isfinite(waiting) or not 0 <= waiting <= 3600:
        parser.error("wait must be finite and in 0..3600 seconds")

    def request(path, body=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{args.port}/api/research/jobs{path}",
            data=None if body is None else canonical(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)

    try:
        if args.command in {"validate", "submit"}:
            if args.spec.stat().st_size > 256 * 1024:
                raise ValueError("Spec exceeds 256 KiB")
            spec = experiment_spec(json.loads(args.spec.read_text()))
            if args.command == "validate":
                print(json.dumps({"spec_hash": digest(spec), "spec": spec}, indent=2))
                return 0
            report = request("", spec)
        else:
            path = "/" + urllib.parse.quote(args.job_id, safe="")
            if args.command == "status":
                report = request(path)
            else:
                report = request(path + "/" + args.command, {})
        if waiting:
            path = "/" + urllib.parse.quote(report["id"], safe="")
            end = time.monotonic() + waiting
            try:
                while report["state"] in {"queued", "running", "canceling"} and time.monotonic() < end:
                    time.sleep(min(.25, max(0, end - time.monotonic())))
                    report = request(path)
            except KeyboardInterrupt:
                report = request(path + "/cancel", {})
                print(json.dumps(report, indent=2))
                return 130
        print(json.dumps(report, indent=2))
        return 0
    except urllib.error.HTTPError as exc:
        print(f"Synapse HTTP {exc.code}: {exc.read().decode(errors='replace')}", file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError, urllib.error.URLError) as exc:
        print(f"Experiment request failed: {exc}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
