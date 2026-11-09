#!/usr/bin/env python3
"""Refresh an explicitly partial P versus P+R chart from measured native trials.

Before P+R has trials, the chart shows measured P progress and an empty P+R slot.
Once both have trials, it compares only identical, verified native starts.
"""

from __future__ import annotations

import argparse
from html import escape
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from compare_p_r import checked, compare, svg


def fetch(path: str, args: argparse.Namespace) -> dict | None:
    if args.ssh_host:
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
        if args.identity:
            command += ["-i", str(args.identity)]
        if args.port:
            command += ["-p", str(args.port)]
        command += [args.ssh_host, f"cat {shlex.quote(path)}"]
        result = subprocess.run(command, text=True, capture_output=True, timeout=20)
        if result.returncode:
            return None
        value = json.loads(result.stdout)
    else:
        source = Path(path)
        if not source.exists():
            return None
        value = json.loads(source.read_text())
    if not isinstance(value.get("trials"), list) or value.get("completed_trials") != len(value["trials"]):
        raise ValueError(f"invalid or partially written summary: {path}")
    return value


def validate_measured(data: dict, label: str) -> None:
    rows = data.get("trials", [])
    if len({row.get("trial_id") for row in rows}) != len(rows):
        raise ValueError(f"{label}: duplicate trial IDs")
    for row in rows:
        if (type(row.get("success")) is not bool
                or row.get("success_predicate_verified") is not True
                or row.get("success_predicate") != "native_LIBERO_check_success"
                or row.get("control_source") != "policy"
                or row.get("contains_human_intervention") is not False
                or row.get("contains_reset_during_rollout") is not False):
            raise ValueError(f"{label}: unverified trial {row.get('trial_id')}")
        for key in ("initial_native_state_sha256", "initial_xml_sha256", "trace_sha256"):
            if not isinstance(row.get(key), str) or len(row[key]) != 64:
                raise ValueError(f"{label}: {row.get('trial_id')} missing {key}")


def pending_svg(data: dict) -> str:
    n = len(data["trials"])
    expected = data.get("expected_trials", "?")
    wins = sum(row["success"] for row in data["trials"])
    rate = wins / n if n else 0
    height = rate * 220
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450" '
            f'viewBox="0 0 800 450"><rect width="800" height="450" fill="#0b1d31"/>'
            f'<g font-family="Arial,sans-serif" fill="#f8fbff">'
            f'<text x="40" y="42" font-size="26" font-weight="bold">Native LIBERO success · interim</text>'
            f'<text x="40" y="69" font-size="15">P: {n}/{escape(str(expected))} completed · P+R: awaiting measured trials</text>'
            f'<path d="M100 315H700" stroke="#d8e2eb"/>'
            f'<rect x="190" y="{315-height:.1f}" width="130" height="{height:.1f}" rx="5" fill="#5c9ec2"/>'
            f'<rect x="460" y="95" width="130" height="220" rx="5" fill="none" stroke="#657a8d" stroke-dasharray="7 7"/>'
            f'<text x="255" y="344" text-anchor="middle" font-size="18">P</text>'
            f'<text x="525" y="344" text-anchor="middle" font-size="18">P + R</text>'
            f'<text x="255" y="373" text-anchor="middle" font-size="18" font-weight="bold">{wins}/{n} · {rate*100:.0f}%</text>'
            f'<text x="525" y="373" text-anchor="middle" font-size="18">pending</text>'
            f'<text x="40" y="423" font-size="13">P-only progress; no P+R comparison is implied.</text>'
            f'</g></svg>')


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(value)
    os.replace(pending, path)


def refresh(args: argparse.Namespace) -> dict:
    p = fetch(args.p, args)
    if p is None:
        raise FileNotFoundError(f"P summary not available: {args.p}")
    validate_measured(p, "P")
    pr = fetch(args.pr, args)
    if pr is None or not pr.get("trials"):
        result = {"schema": "rg.measured_p_vs_p_plus_r.live.v1", "status": "P_only_partial",
                  "P": {"completed": len(p["trials"]), "expected": p.get("expected_trials"),
                        "successes": sum(row["success"] for row in p["trials"])},
                  "P_plus_R": {"completed": 0, "status": "awaiting_measured_trials"},
                  "claim_scope": "P-only native interim progress; no P+R comparison"}
        picture = pending_svg(p)
    else:
        validate_measured(pr, "P+R")
        result = compare(p, pr)
        result["P_completed"] = len(p["trials"])
        result["P_plus_R_completed"] = len(pr["trials"])
        result["expected_trials"] = p.get("expected_trials")
        picture = svg(result)
    atomic_text(args.output / "comparison.json", json.dumps(result, indent=2) + "\n")
    atomic_text(args.output / "comparison.svg", picture)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--P", dest="p", required=True, help="P summary.json path")
    parser.add_argument("--P-plus-R", dest="pr", required=True, help="P+R summary.json path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ssh-host", help="optional SSH host to read both remote paths")
    parser.add_argument("--identity", type=Path, help="SSH identity for --ssh-host")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--interval", type=int, default=30, help="seconds between checks; 0 for once")
    args = parser.parse_args()
    while True:
        try:
            result = refresh(args)
            print(json.dumps({"status": result["status"], "paired_trials": result.get("paired_trials"),
                              "P": result["P"], "P_plus_R": result["P_plus_R"]}), flush=True)
        except (FileNotFoundError, ValueError, subprocess.TimeoutExpired) as error:
            print(json.dumps({"status": "waiting", "error": str(error)}), flush=True)
            if args.interval == 0:
                raise
        if args.interval == 0:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
