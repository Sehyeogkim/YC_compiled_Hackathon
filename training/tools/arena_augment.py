"""Make five traceable temporal windows per annotated Physical Arena segment.

This preserves recorded state/action pairs. It does not create new rollouts,
new successes, or training-ready RGB observations.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def positions(start: int, end: int) -> list[tuple[int, int]]:
    length = end - start
    if length < 20:
        return []
    width = min(length - 4, max(16, math.ceil(length * .6)))
    travel = length - width
    starts = [start + round(travel * index / 4) for index in range(5)]
    if len(set(starts)) != 5:
        return []
    return [(first, first + width) for first in starts]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path, help="Bulk full-episode JSONL(.gz) export")
    parser.add_argument("segments", type=Path, help="arena_segment.py JSON manifest")
    parser.add_argument("--output", type=Path, required=True, help="Window JSONL.gz output")
    parser.add_argument("--manifest", type=Path, required=True, help="Lineage/count JSON output")
    args = parser.parse_args()
    episodes = read_jsonl(args.episodes)
    annotations = json.loads(args.segments.read_text(encoding="utf-8"))["annotations"]
    by_id = {item["episode_id"]: item for item in annotations}
    seen: set[str] = set()
    windows: list[dict] = []
    skipped: list[dict] = []
    for episode in episodes:
        episode_id = episode.get("id")
        if episode_id in seen:
            raise ValueError(f"duplicate parent episode: {episode_id}")
        seen.add(episode_id)
        if episode.get("schema") != "wemine.physical-arena.episode.v1":
            raise ValueError(f"unexpected schema for {episode_id}")
        annotation = by_id.get(episode_id)
        if annotation is None:
            raise ValueError(f"missing segment annotation for {episode_id}")
        steps = episode.get("steps")
        if not isinstance(steps, list) or len(steps) != annotation["total_steps"]:
            raise ValueError(f"step count mismatch for {episode_id}")
        for kind, boundary in (("recovery_candidate", annotation.get("recovery_candidate")),
                               ("task_execution", annotation.get("task_execution"))):
            if boundary is None:
                continue
            start, end = boundary["start_step"], boundary["end_step"]
            if kind == "recovery_candidate":
                # R′ starts after the failed grasp is released; the failed
                # attempt itself is context, not a recovery action target.
                start = boundary.get("release_step", start)
            if kind == "recovery_candidate" and start >= end:
                skipped.append({"episode_id": episode_id, "segment": kind, "reason": "no_post_release_recovery_steps"})
                continue
            if not (0 <= start < end <= len(steps)):
                raise ValueError(f"invalid {kind} boundary for {episode_id}")
            if kind == "task_execution" and not episode.get("result", {}).get("native_success"):
                skipped.append({"episode_id": episode_id, "segment": kind, "reason": "parent_not_successful"})
                continue
            slices = positions(start, end)
            if len(slices) != 5:
                skipped.append({"episode_id": episode_id, "segment": kind, "reason": "segment_under_20_steps"})
                continue
            for index, (first, last) in enumerate(slices):
                windows.append({
                    "schema": "rg.arena_temporal_window.v1",
                    "id": f"{episode_id}:{kind}:{index}",
                    "parent_episode_id": episode_id,
                    "split_group": episode_id,
                    "source": episode.get("source"),
                    "task_id": episode.get("task_id"),
                    "segment": kind,
                    "segment_status": boundary.get("status"),
                    "window_index": index,
                    "source_start_step": first,
                    "source_end_step_exclusive": last,
                    "source_total_steps": len(steps),
                    "model_sha256": episode.get("model_sha256"),
                    "manifest_sha256": episode.get("manifest_sha256"),
                    "initial_state": steps[first]["state"],
                    "steps": steps[first:last],
                    "parent_native_success": bool(episode.get("result", {}).get("native_success")),
                    "independent_rollout": False,
                    "policy_failure_provenance": False,
                    "training_eligible": False,
                    "replay_status": "requires_native_action_replay_and_rgb",
                    "augmentation": "overlapping_temporal_crop_only",
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as dest:
        for window in windows:
            dest.write(json.dumps(window, separators=(",", ":")) + "\n")
    counts = {
        "schema": "rg.arena_augmentation_manifest.v1",
        "source_file": str(args.episodes),
        "annotation_file": str(args.segments),
        "output_file": str(args.output),
        "original_episode_count": len(episodes),
        "window_count": len(windows),
        "recovery_candidate_windows": sum(w["segment"] == "recovery_candidate" for w in windows),
        "recovery_candidate_windows_from_successful_parents": sum(w["segment"] == "recovery_candidate" and w["parent_native_success"] for w in windows),
        "recovery_candidate_windows_from_incomplete_parents": sum(w["segment"] == "recovery_candidate" and not w["parent_native_success"] for w in windows),
        "task_execution_windows": sum(w["segment"] == "task_execution" for w in windows),
        "distinct_parent_episodes": len({w["parent_episode_id"] for w in windows}),
        "fivefold_means": "five overlapping temporal windows per qualifying original segment",
        "independent_rollouts_created": 0,
        "evaluation_split_unit": "parent_episode_id",
        "training_eligible": False,
        "skipped": skipped,
    }
    args.manifest.write_text(json.dumps(counts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
