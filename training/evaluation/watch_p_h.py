#!/usr/bin/env python3
"""Live native P versus P+H chart for human task teleoperation demos.

The source checkpoint's older P+R file naming is preserved in provenance, but
these full task demonstrations are H, with no qualified R-prime recovery data.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import time

from PIL import Image, ImageDraw, ImageFont

from compare_p_r import compare
from watch_p_r import atomic_text, fetch, validate_measured


def measured_result(p: dict, h: dict | None) -> dict:
    if h is None or not h.get("trials"):
        rows = p["trials"]
        return {
            "schema": "rg.measured_p_vs_p_plus_h.v1",
            "status": "P_only_H_pending",
            "data_kind": "human_task_teleoperation",
            "recovery_data_present": False,
            "paired_trials": 0,
            "expected_trials": p.get("expected_trials"),
            "P": {"completed": len(rows), "successes": sum(r["success"] for r in rows),
                  "rate": sum(r["success"] for r in rows) / len(rows) if rows else None,
                  "policy": p.get("policy")},
            "P_plus_H": {"completed": 0, "status": "awaiting_measured_trials"},
            "claim_scope": "P native results only; no P+H comparison yet",
        }
    validate_measured(h, "P+H")
    paired = compare(p, h)
    h_result = paired.pop("P_plus_R")
    outcomes = paired.pop("paired_outcomes_P_then_P_plus_R")
    raw_policy = h_result.pop("policy")
    h_result["checkpoint_provenance"] = raw_policy
    h_result["data_kind"] = "human_task_teleoperation"
    return {
        **paired,
        "schema": "rg.measured_p_vs_p_plus_h.v1",
        "data_kind": "human_task_teleoperation",
        "recovery_data_present": False,
        "expected_trials": p.get("expected_trials"),
        "P_completed": len(p["trials"]),
        "P_plus_H_completed": len(h["trials"]),
        "P_plus_H": h_result,
        "paired_outcomes_P_then_P_plus_H": outcomes,
        "claim_scope": "native LIBERO task success on identical measured starts; H is human full-task teleoperation, not recovery R or R-prime",
    }


def svg(result: dict) -> str:
    n = result["paired_trials"]
    paired = n > 0
    p = result["P"]
    h = result["P_plus_H"]
    p_n = n if paired else p["completed"]
    h_n = n if paired else 0
    p_rate = p["rate"] or 0
    h_rate = h.get("rate") or 0
    p_height = p_rate * 220
    h_height = h_rate * 220
    status = f"{n}/{result['expected_trials']} paired starts · interim" if paired else "P+H evaluation pending"
    second = (f'<rect x="460" y="{315-h_height:.1f}" width="130" height="{h_height:.1f}" rx="5" fill="#f3a714"/>'
              f'<text x="525" y="373" text-anchor="middle" font-size="18" font-weight="bold">{h["successes"]}/{h_n} · {h_rate*100:.0f}%</text>') if paired else (
              '<rect x="460" y="95" width="130" height="220" rx="5" fill="none" stroke="#657a8d" stroke-dasharray="7 7"/>'
              '<text x="525" y="373" text-anchor="middle" font-size="18">pending</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450" viewBox="0 0 800 450">'
            f'<rect width="800" height="450" fill="#0b1d31"/>'
            f'<g font-family="Arial,sans-serif" fill="#f8fbff">'
            f'<text x="40" y="42" font-size="26" font-weight="bold">P vs P + H · native task success</text>'
            f'<text x="40" y="70" font-size="15">{status}</text>'
            f'<path d="M100 315H700" stroke="#d8e2eb"/>'
            f'<rect x="190" y="{315-p_height:.1f}" width="130" height="{p_height:.1f}" rx="5" fill="#5c9ec2"/>'
            f'{second}'
            f'<text x="255" y="344" text-anchor="middle" font-size="18">P</text>'
            f'<text x="525" y="344" text-anchor="middle" font-size="18">P + H</text>'
            f'<text x="255" y="373" text-anchor="middle" font-size="18" font-weight="bold">{p["successes"]}/{p_n} · {p_rate*100:.0f}%</text>'
            f'<text x="40" y="420" font-size="13">H = full human task teleoperation · no recovery R′ data · paired partial results</text>'
            f'</g></svg>')


def font(size: int):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def png(result: dict, *, public: bool = False) -> bytes:
    canvas = Image.new("RGB", (800, 450), "#0b1d31")
    draw = ImageDraw.Draw(canvas)
    title = "P vs P + public H · native task success" if public else "P vs P + H · native task success"
    draw.text((40, 23), title, fill="#f8fbff", font=font(26))
    n = result["paired_trials"]
    paired = n > 0
    caption = f"{n}/{result['expected_trials']} paired starts · interim" if paired else "P+H evaluation pending"
    draw.text((40, 63), caption, fill="#f8fbff", font=font(15))
    draw.line((100, 315, 700, 315), fill="#d8e2eb", width=1)
    for key, x, color in (("P", 190, "#5c9ec2"), ("P_plus_H", 460, "#f3a714")):
        row = result[key]
        if key == "P_plus_H" and not paired:
            draw.rectangle((x, 95, x + 130, 315), outline="#657a8d", width=2)
            label = "pending"
        else:
            rate = row["rate"] or 0
            draw.rounded_rectangle((x, 315 - rate * 220, x + 130, 315), radius=5, fill=color)
            label = f"{row['successes']}/{n if paired else row['completed']} · {rate*100:.0f}%"
        draw.text((x + 65, 335), "P" if key == "P" else "P + H", fill="#f8fbff", font=font(18), anchor="mt")
        draw.text((x + 65, 365), label, fill="#f8fbff", font=font(18), anchor="mt")
    caption = ("Public H = LIBERO human task teleoperation · no recovery R′ data · paired partial results"
               if public else "H = full human task teleoperation · no recovery R′ data · paired partial results")
    draw.text((40, 408), caption,
              fill="#f8fbff", font=font(12))
    output = BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def refresh(args: argparse.Namespace) -> dict:
    p = fetch(args.p, args)
    if p is None:
        raise FileNotFoundError(f"P summary not available: {args.p}")
    validate_measured(p, "P")
    h = fetch(args.h, args)
    result = measured_result(p, h)
    atomic_text(args.output / "comparison.json", json.dumps(result, indent=2) + "\n")
    atomic_text(args.output / "comparison.svg", svg(result))
    atomic_bytes(args.output / "comparison.png", png(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--P", dest="p", required=True)
    parser.add_argument("--P-plus-H", dest="h", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ssh-host")
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    while True:
        try:
            result = refresh(args)
            print(json.dumps({"status": result["status"], "paired_trials": result["paired_trials"],
                              "P": result["P"], "P_plus_H": result["P_plus_H"]}), flush=True)
        except (FileNotFoundError, ValueError, subprocess.TimeoutExpired) as error:
            print(json.dumps({"status": "waiting", "error": str(error)}), flush=True)
            if args.interval == 0:
                raise
        if args.interval == 0:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
