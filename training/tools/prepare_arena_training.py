#!/usr/bin/env python3
"""Stage Physical Arena human runs for review and later native replay.

This does not produce a LeRobot dataset. Browser runs lack the two aligned
policy cameras and verified native policy state/action contract. Five source-
linked temporal windows can be made separately without changing robot states.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from arena_segment import annotate, read_episodes


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def write_segment(path: Path, meta: dict, steps: list[dict], start: int, end: int,
                  kind: str, annotation: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    segment_meta = {
        "schema": "rg.arena_staged_segment.v1",
        "kind": kind,
        "source_episode_id": meta["id"],
        "source": meta.get("source"),
        "task_id": meta.get("task_id"),
        "source_model_sha256": meta.get("model_sha256"),
        "source_control_schema": meta.get("controls"),
        "source_start_step": start,
        "source_end_step": end,
        "segment_steps": end - start,
        "training_eligible": False,
        "replay_status": "requires_native_action_replay_and_two_policy_cameras",
        "annotation_status": annotation.get("review_status"),
        "augmentation_status": "pending_separate_temporal_window_step",
    }
    with gzip.open(path, "wt", encoding="utf-8") as output:
        output.write(json.dumps({"type": "metadata", **segment_meta}, separators=(",", ":")) + "\n")
        for original_step, step in enumerate(steps[start:end], start):
            output.write(json.dumps({"type": "step", "source_step": original_step,
                                     **step}, separators=(",", ":")) + "\n")


def stage(source: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError(f"output exists: {output_dir}; choose a fresh directory")
    episodes = read_episodes(source)
    ids: set[str] = set()
    rows: list[dict] = []
    exclusions: list[dict] = []
    output_dir.mkdir(parents=True)
    for meta, steps in episodes:
        eid = meta.get("id")
        if not isinstance(eid, str) or not eid or eid in ids:
            raise ValueError(f"missing or duplicate episode ID: {eid!r}")
        ids.add(eid)
        if meta.get("source") != "human_browser_unverified":
            exclusions.append({"episode_id": eid, "reason": "non_human_browser_source"})
            continue
        if not steps or any(len(step.get("action", [])) != 7 or not isinstance(step.get("state", {}).get("qpos"), list)
                            for step in steps):
            exclusions.append({"episode_id": eid, "reason": "incomplete_state_action_sequence"})
            continue
        ann = annotate(meta, steps)
        successful = bool(meta.get("result", {}).get("native_success"))
        recovery = ann.get("recovery_candidate")
        if recovery:
            # 3′ starts once the failed grasp has been released. The original
            # failed attempt remains in the untouched source, outside R′.
            start, end = recovery["release_step"], recovery["end_step"]
            if end > start:
                relative = f"segments/R_prime_candidate/{eid}.jsonl.gz"
                write_segment(output_dir / relative, meta, steps, start, end,
                              "R_prime_candidate", ann)
                rows.append({"episode_id": eid, "kind": "R_prime_candidate",
                             "path": relative, "start_step": start, "end_step": end,
                             "steps": end - start, "training_eligible": False,
                             "failure_source": "human_self_failure_not_policy_failure"})
        if successful:
            task = ann.get("task_execution") or {}
            start, end = int(task.get("start_step", 0)), len(steps)
            if end > start:
                relative = f"segments/R/{eid}.jsonl.gz"
                write_segment(output_dir / relative, meta, steps, start, end, "R", ann)
                rows.append({"episode_id": eid, "kind": "R", "path": relative,
                             "start_step": start, "end_step": end,
                             "steps": end - start, "training_eligible": False})
        else:
            exclusions.append({"episode_id": eid, "reason": "native_task_not_successful"})
    manifest = {
        "schema": "rg.arena_training_staging.v1",
        "source_file": str(source.resolve()),
        "source_sha256": sha256(source),
        "source_episodes": len(episodes),
        "staged_segments": rows,
        "excluded_from_task_training": exclusions,
        "augmentation_factor_requested": 5,
        "augmentation_unit": "five source-linked temporal windows per original segment; not five independent source episodes",
        "augmentation_applied": False,
        "training_eligible": False,
        "requirements_before_smolvla": [
            "sequential native LIBERO action replay from verified matching start state",
            "pre-action 8D policy state and 7D executed native action alignment",
            "two synchronized policy RGB camera frames per control step",
            "held-out replay QA and recovery boundary review",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="bulk Physical Arena .jsonl.gz export")
    parser.add_argument("output_dir", type=Path, help="new staging directory")
    args = parser.parse_args()
    result = stage(args.source, args.output_dir)
    print(json.dumps({"source_episodes": result["source_episodes"],
                      "task_segments": sum(row["kind"] == "R" for row in result["staged_segments"]),
                      "recovery_candidates": sum(row["kind"] == "R_prime_candidate" for row in result["staged_segments"]),
                      "augmentation_applied": False,
                      "training_eligible": False,
                      "manifest": str(args.output_dir / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
