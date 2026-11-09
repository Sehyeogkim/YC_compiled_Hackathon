"""Annotate Physical Arena can-to-basket runs without changing their raw provenance.

Only an observed failed human grasp can create a *candidate* recovery interval.
The result is an annotation for review, never a qualified VLA training segment.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

TASK = "libero_object__pick_up_the_alphabet_soup_and_place_it_in_the_basket"
# Verified against WEMINE's pinned task manifest: alphabet_soup_1_joint0 has
# qpos_adr=9; basket_1_joint0 has qpos_adr=16. Both are free joints, so qpos
# [adr:adr+3] holds each body's XYZ translation in the exported MuJoCo state.
CAN_QPOS = 9
PINNED_MODEL_SHA256 = "84fb57189d55cf42e645ebb1e4463151ccc6356915e9d42e9bbf624bf36d00c1"


def validate(meta: dict, path: Path) -> None:
    if meta.get("schema") != "wemine.physical-arena.episode.v1" or meta.get("task_id") != TASK:
        raise ValueError(f"{path}: incompatible episode or task")
    if meta.get("model_sha256") != PINNED_MODEL_SHA256:
        raise ValueError(f"{path}: scene model hash differs from pinned can task")


def read_episodes(path: Path) -> list[tuple[dict, list[dict]]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"{path}: empty JSONL")
    if all(row.get("schema") == "wemine.physical-arena.episode.v1" and isinstance(row.get("steps"), list)
           for row in rows):
        # arena-export.html writes one complete IndexedDB episode per line.
        out = []
        for episode in rows:
            meta = {key: value for key, value in episode.items() if key != "steps"}
            validate(meta, path)
            out.append((meta, episode["steps"]))
        return out
    # Older per-run download writes metadata followed by individual step rows.
    if rows[0].get("type") == "metadata":
        meta = {key: value for key, value in rows[0].items() if key != "type"}
        validate(meta, path)
        steps = [{key: value for key, value in row.items() if key != "type"}
                 for row in rows[1:] if row.get("type") == "step"]
        return [(meta, steps)]
    raise ValueError(f"{path}: expected full-episode or metadata/step Physical Arena JSONL")


def xyz(step: dict, offset: int) -> list[float]:
    values = step["state"]["qpos"][offset:offset + 3]
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError("missing finite object position")
    return values


def target(step: dict) -> list[float] | None:
    values = step.get("input", {}).get("target")
    if not isinstance(values, list) or len(values) != 3:
        return None
    return values if all(isinstance(v, (int, float)) and math.isfinite(v) for v in values) else None


def grip(step: dict) -> bool:
    return bool(step.get("input", {}).get("grip_closed"))


def near_can(step: dict, xy: float, dz: float) -> bool:
    hand = target(step)
    if hand is None:
        return False
    can = xyz(step, CAN_QPOS)
    return math.hypot(hand[0] - can[0], hand[1] - can[1]) <= xy and abs(hand[2] - can[2]) <= dz


def annotate(meta: dict, steps: list[dict]) -> dict:
    base = {
        "schema": "rg.arena_segment_annotation.v1",
        "episode_id": meta.get("id"),
        "task_id": TASK,
        "source": meta.get("source"),
        "native_success": bool(meta.get("result", {}).get("native_success")),
        "training_eligible": False,
        "replay_status": "requires_native_action_replay",
        "policy_failure_provenance": False,
        "review_status": "unverified_heuristic",
        "total_steps": len(steps),
    }
    if not steps:
        return {**base, "recovery_candidate": None, "task_execution": None, "reason": "empty_episode"}

    # Each rising gripper-close near the can is a grasp attempt. An attempt is
    # failed only if the can never rises by 2 cm before the hand opens again.
    failures: list[tuple[int, int]] = []
    for i in range(1, len(steps)):
        if grip(steps[i]) and not grip(steps[i - 1]) and near_can(steps[i], .12, .16):
            opened = next((j for j in range(i + 1, len(steps)) if not grip(steps[j])), None)
            if opened is None:
                continue
            z0 = xyz(steps[i], CAN_QPOS)[2]
            if max(xyz(step, CAN_QPOS)[2] for step in steps[i:opened + 1]) < z0 + .02:
                failures.append((i, opened))

    if not failures:
        return {**base, "recovery_candidate": None,
                "task_execution": {"start_step": 0, "end_step": len(steps), "status": "whole_run"},
                "reason": "no_observed_failed_grasp"}

    failed_at, opened_at = failures[-1]
    failed_hand = target(steps[failed_at])
    ready = None
    for j in range(opened_at, len(steps)):
        hand = target(steps[j])
        if hand is None or grip(steps[j]):
            continue
        can = xyz(steps[j], CAN_QPOS)
        withdrawn = failed_hand is not None and (hand[2] >= failed_hand[2] + .04 or
                     math.hypot(hand[0] - failed_hand[0], hand[1] - failed_hand[1]) >= .05)
        above_can = math.hypot(hand[0] - can[0], hand[1] - can[1]) <= .10 and hand[2] >= can[2] + .06
        if withdrawn and above_can:
            ready = j
            break
    if ready is None:
        return {**base, "recovery_candidate": None,
                "task_execution": {"start_step": 0, "end_step": len(steps), "status": "unsplit"},
                "reason": "failed_grasp_seen_but_no_safe_pregrasp_boundary",
                "failed_grasp_step": failed_at}

    return {**base,
            "recovery_candidate": {"start_step": failed_at, "end_step": ready,
                                   "status": "human_self_failure_candidate", "failed_grasp_step": failed_at,
                                   "release_step": opened_at, "boundary_rule": "open_gripper_withdraw_then_above_can"},
            "task_execution": {"start_step": ready, "end_step": len(steps),
                               "status": "candidate_after_recovery"},
            "reason": "failed_grasp_and_safe_pregrasp_found"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path, help="Physical Arena .jsonl or .jsonl.gz files")
    parser.add_argument("--output", type=Path, help="Write annotation JSON here (default: stdout)")
    args = parser.parse_args()
    annotations = []
    for path in args.episodes:
        for meta, steps in read_episodes(path):
            annotations.append({"source_file": str(path), **annotate(meta, steps)})
    result = {"schema": "rg.arena_segment_manifest.v1", "annotations": annotations}
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
