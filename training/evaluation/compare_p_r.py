#!/usr/bin/env python3
"""Draw a P versus P+R chart from measured, paired native LIBERO trials only."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path


def checked(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data.get("trials"), list) or not data["trials"]:
        raise ValueError(f"{path}: no measured trials")
    if data.get("completed_trials") != len(data["trials"]):
        raise ValueError(f"{path}: trial count mismatch")
    ids = set()
    for row in data["trials"]:
        trial = row.get("trial_id")
        if not isinstance(trial, str) or trial in ids:
            raise ValueError(f"{path}: missing or duplicate trial ID")
        ids.add(trial)
        if (type(row.get("success")) is not bool
                or row.get("success_predicate_verified") is not True
                or row.get("success_predicate") != "native_LIBERO_check_success"
                or row.get("control_source") != "policy"
                or row.get("contains_human_intervention") is not False
                or row.get("contains_reset_during_rollout") is not False):
            raise ValueError(f"{path}: trial {trial} lacks autonomous native success evidence")
        for key in ("initial_native_state_sha256", "initial_xml_sha256", "trace_sha256"):
            if not isinstance(row.get(key), str) or len(row[key]) != 64:
                raise ValueError(f"{path}: trial {trial} missing {key}")
    return data


def compare(p: dict, pr: dict) -> dict:
    first = {row["trial_id"]: row for row in p["trials"]}
    second = {row["trial_id"]: row for row in pr["trials"]}
    common = sorted(first.keys() & second.keys())
    if not common:
        raise ValueError("no paired native trials")
    pairs = []
    for trial_id in common:
        a, b = first[trial_id], second[trial_id]
        for key in ("initial_state_id", "policy_rng_seed", "env_seed", "suite", "task_id",
                    "initial_native_state_sha256", "initial_xml_sha256", "initial_objects"):
            if a.get(key) != b.get(key):
                raise ValueError(f"{trial_id}: unmatched {key}")
        pairs.append((a, b))
    n = len(pairs)
    wins_p = sum(a["success"] for a, _ in pairs)
    wins_pr = sum(b["success"] for _, b in pairs)
    outcomes = {key: 0 for key in ("00", "01", "10", "11")}
    for a, b in pairs:
        outcomes[f'{int(a["success"])}{int(b["success"])}'] += 1
    return {
        "schema": "rg.measured_p_vs_p_plus_r.v1",
        "status": "complete" if p.get("complete") and pr.get("complete") and len(first) == len(second) == n else "paired_partial",
        "paired_trials": n,
        "P": {"successes": wins_p, "rate": wins_p / n, "policy": p["trials"][0].get("policy")},
        "P_plus_R": {"successes": wins_pr, "rate": wins_pr / n, "policy": pr["trials"][0].get("policy")},
        "paired_outcomes_P_then_P_plus_R": outcomes,
        "trial_ids": common,
        "claim_scope": "native LIBERO paired task success on measured trials only; no recovery-policy evaluation",
    }


def svg(result: dict) -> str:
    count = result["paired_trials"]
    title = "Native LIBERO task success" + (" · partial" if result["status"] != "complete" else "")
    bars = []
    for key, label, x, color in (("P", "P", 190, "#5c9ec2"),
                                 ("P_plus_R", "P + R", 460, "#f3a714")):
        row = result[key]
        h = row["rate"] * 220
        bars.append(f'<rect x="{x}" y="{315-h:.1f}" width="130" height="{h:.1f}" rx="5" fill="{color}"/>')
        bars.append(f'<text x="{x+65}" y="344" text-anchor="middle" font-size="18">{escape(label)}</text>')
        bars.append(f'<text x="{x+65}" y="373" text-anchor="middle" font-size="18" font-weight="bold">{row["successes"]}/{count} · {row["rate"]*100:.0f}%</text>')
    grid = "".join(f'<path d="M100 {315-i*55}H700" stroke="#d8e2eb"/><text x="88" y="{320-i*55}" text-anchor="end" font-size="12">{i*25}%</text>' for i in range(5))
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450" viewBox="0 0 800 450"><rect width="800" height="450" fill="#0b1d31"/><g font-family="Arial,sans-serif" fill="#f8fbff"><text x="40" y="42" font-size="26" font-weight="bold">{escape(title)}</text><text x="40" y="69" font-size="15">{count} paired starts · native success predicate</text>{grid}{"".join(bars)}<text x="40" y="423" font-size="13">R: human task continuation data · paired outcomes verified; partial results labeled.</text></g></svg>'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--P", type=Path, required=True, help="P native evaluation summary.json")
    parser.add_argument("--P-plus-R", dest="pr", type=Path, required=True, help="P+R native evaluation summary.json")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; use a fresh directory")
    result = compare(checked(args.P), checked(args.pr))
    args.output.mkdir(parents=True)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "comparison.svg").write_text(svg(result))
    print(json.dumps({"status": result["status"], "paired_trials": result["paired_trials"],
                      "P": result["P"], "P_plus_R": result["P_plus_R"]}, indent=2))


if __name__ == "__main__":
    main()
