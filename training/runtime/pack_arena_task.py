#!/usr/bin/env python3
"""Pack verified native replays of human Physical Arena R into LeRobot at 20 Hz.

Run with WEMINE's .venv-lerobot Python. This refuses browser-only data and
preserves one LeRobot episode per distinct human source episode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

TASK = "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
SOURCE_TASK = f"libero_object__{TASK}"
WEMINE_ROOT = Path("/Users/jeff/project/WEMINE/game1016")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="rg/arena_can_R_native")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.root.exists():
        parser.error("output exists; choose a new dataset path")
    if not args.source_bundle.is_file():
        parser.error("missing original browser export")
    candidates = sorted(args.frames_root.glob("physical-arena/G/*"))
    if not candidates:
        parser.error("no native rendered episode folders")
    folders: list[Path] = []
    reports: list[dict] = []
    ids: set[str] = set()
    for folder in candidates:
        report_file, frames_file = folder / "episode.json", folder / "frames.npz"
        if not report_file.is_file():
            continue  # bulk render may still be writing this candidate
        report = json.loads(report_file.read_text())
        if report.get("eligible") is not True:
            continue
        if not frames_file.is_file():
            parser.error(f"eligible native episode is missing frames: {folder}")
        episode_id = report.get("episode_id")
        if (report.get("schema") != "wemine.lerobot_frames.v1"
                or report.get("game") != "physical-arena" or report.get("source") != "G"
                or report.get("task") != TASK or report.get("human") is not True
                or report.get("eligible") is not True or report.get("reasons") != []
                or report.get("native_success_step") is None
                or report.get("browser_success_step") is None
                or episode_id in ids or not isinstance(episode_id, str)):
            parser.error(f"invalid human/native replay provenance in {folder}")
        ids.add(episode_id)
        folders.append(folder)
        with np.load(frames_file) as frames:
            counts = [len(frames[key]) for key in ("image", "image2", "state", "action")]
            if len(set(counts)) != 1 or counts[0] < 2:
                parser.error(f"misaligned frames in {folder}: {counts}")
            if (frames["image"].shape[1:] != (256, 256, 3)
                    or frames["image2"].shape[1:] != (256, 256, 3)
                    or frames["state"].shape[1:] != (8,)
                    or frames["action"].shape[1:] != (7,)
                    or frames["image"].dtype != np.uint8 or frames["image2"].dtype != np.uint8
                    or not np.isfinite(frames["state"]).all()
                    or not np.isfinite(frames["action"]).all()):
                parser.error(f"invalid policy observation/action array in {folder}")
            reports.append({"episode_id": episode_id, "frames": counts[0],
                            "start_state_sha256": report["start_state_sha256"],
                            "report_sha256": sha256(report_file), "frames_sha256": sha256(frames_file)})
    if not folders:
        parser.error("no eligible human/native success episodes")
    unique_starts = len({row["start_state_sha256"] for row in reports})

    args.root.parent.mkdir(parents=True, exist_ok=True)
    episode_list = args.root.parent / f"{args.root.name}.episodes.txt"
    episode_list.write_text("\n".join(str(path.resolve()) for path in folders) + "\n")
    sys.path.insert(0, str(WEMINE_ROOT))
    from lerobot_export import pack
    pack.FPS = 20  # WEMINE default metadata is 10 although these actions are 20 Hz.
    command = [str(episode_list), "--repo-id", args.repo_id, "--root", str(args.root)]
    if args.video:
        command.append("--video")
    pack.main(command)
    info = json.loads((args.root / "meta/info.json").read_text())
    if info.get("fps") != 20 or info.get("total_episodes") != len(reports):
        raise RuntimeError("packed LeRobot metadata failed FPS/episode check")
    provenance = {
        "schema": "rg.arena_native_task_dataset.v1", "task_id": SOURCE_TASK,
        "source": "human_browser_unverified", "control_source": "human_game_play",
        "source_bundle": str(args.source_bundle.resolve()), "source_bundle_sha256": sha256(args.source_bundle),
        "parent_episodes": reports, "augmentation": "none_full_verified_episodes",
        "unique_initial_states": unique_starts,
        "fps": 20, "state_dim": 8, "action_dim": 7,
        "camera_keys": ["observation.images.image", "observation.images.image2"],
        "native_replay_verified": True, "training_eligible": True,
        "policy_failure_provenance": False, "recovery_data": False,
        "held_out_native_start_ids": list(range(25, 50)),
        "eval_parent_ids": [],
    }
    (args.root / "RG_ARENA_NATIVE_REPLAY.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"dataset": str(args.root), "episodes": len(reports),
                      "frames": sum(row["frames"] for row in reports), "fps": 20}))


if __name__ == "__main__":
    main()
