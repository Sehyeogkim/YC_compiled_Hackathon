#!/usr/bin/env python3
"""Live paired P versus PUBLIC LIBERO human-teleoperation fine-tune chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import subprocess

from watch_p_h import atomic_bytes, atomic_text, measured_result, png, svg
from watch_p_r import fetch, validate_measured


def refresh(args: argparse.Namespace) -> dict:
    p = fetch(args.p, args)
    if p is None:
        raise FileNotFoundError(f"P summary not available: {args.p}")
    validate_measured(p, "P")
    h = fetch(args.h, args)
    if h is not None:
        policy = h.get("policy") or {}
        if (h.get("arm") != "P+public-H"
                or policy.get("label") != "P+public-H"
                or not policy.get("finetune_manifest_sha256")):
            raise ValueError("second summary is not the PUBLIC human checkpoint evaluation")
    result = measured_result(p, h)
    result["training_data_source"] = "PUBLIC LIBERO human teleoperation demos"
    result["browser_h_diagnostic_excluded"] = True
    result["recovery_data_present"] = False
    result["claim_scope"] = (
        "Native LIBERO task success on paired measured starts; PUBLIC human full-task "
        "teleoperation fine-tune; no recovery-policy claim"
    )
    # The public-H source is visibly distinguished from the earlier browser-H diagnostic.
    graphic = svg(result).replace("P vs P + H · native task success",
                                  "P vs P + public H · native task success")
    graphic = graphic.replace("H = full human task teleoperation",
                              "Public H = LIBERO human task teleoperation")
    picture = png(result, public=True)
    atomic_text(args.output / "comparison.json", json.dumps(result, indent=2) + "\n")
    atomic_text(args.output / "comparison.svg", graphic)
    atomic_bytes(args.output / "comparison.png", picture)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--P", dest="p", required=True)
    parser.add_argument("--P-plus-public-H", dest="h", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ssh-host")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    while True:
        try:
            result = refresh(args)
            print(json.dumps({"status": result["status"],
                              "paired_trials": result["paired_trials"]}), flush=True)
        except (FileNotFoundError, ValueError, subprocess.TimeoutExpired) as error:
            print(json.dumps({"status": "waiting", "error": str(error)}), flush=True)
            if args.interval == 0:
                raise
        if args.interval == 0:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
