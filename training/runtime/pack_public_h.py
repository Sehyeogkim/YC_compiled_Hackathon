#!/usr/bin/env python3
"""Pack pinned public LIBERO human demos for the pinned P policy at true 20 Hz."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

TASK = "pick_up_the_alphabet_soup_and_place_it_in_the_basket"
SOURCE_REPO = "yifengzhu-hf/LIBERO-datasets"
REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
FEATURES = {
    "observation.images.image": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.image2": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames-root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo-id", default="rg/public_LIBERO_object_task0_H")
    p.add_argument("--demos", nargs="+", type=int, required=True)
    args = p.parse_args()
    if args.root.exists():
        p.error("output already exists")
    demos = sorted(set(args.demos))
    if demos != sorted(args.demos) or not demos or any(i < 0 or i >= 25 for i in demos):
        p.error("demos must be distinct, sorted, and held out from eval IDs25–49")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset.create(args.repo_id, fps=20, features=FEATURES, root=args.root,
                               robot_type="panda", use_videos=True, image_writer_threads=4,
                               vcodec="h264")
    parents = []
    source_manifest = json.loads((args.source_root / "source.json").read_text())
    if source_manifest.get("repo") != SOURCE_REPO or source_manifest.get("revision") != REVISION:
        raise ValueError("public source pin mismatch")
    for i in demos:
        folder = args.frames_root / "claw-machine" / "H" / f"{TASK}__demo_{i}"
        report_path = folder / "episode.json"
        frames_path = folder / "frames.npz"
        report = json.loads(report_path.read_text())
        if (report.get("eligible") is not True or report.get("human") is not True
                or report.get("source") != "H" or report.get("task") != TASK
                or report.get("native_success_step") is None
                or report.get("meta", {}).get("dataset", {}).get("repo") != SOURCE_REPO
                or report.get("meta", {}).get("dataset", {}).get("revision") != REVISION):
            raise ValueError(f"invalid public human/native provenance for demo_{i}")
        source = args.source_root / f"demo_{i}"
        raw_states = np.load(source / "states.npy")
        raw_actions = np.load(source / "actions.npy")
        if len(raw_states) != len(raw_actions) or not np.allclose(np.diff(raw_states[:, 0]), .05, atol=1e-4):
            raise ValueError(f"demo_{i}: raw state/action cadence is not aligned 20Hz")
        with np.load(frames_path) as f:
            sizes = [len(f[k]) for k in ("image", "image2", "state", "action")]
            if len(set(sizes)) != 1 or sizes[0] < 2:
                raise ValueError(f"demo_{i}: frame mismatch {sizes}")
            if (f["image"].shape[1:] != (256, 256, 3) or f["image2"].shape[1:] != (256, 256, 3)
                    or f["state"].shape[1:] != (8,) or f["action"].shape[1:] != (7,)):
                raise ValueError(f"demo_{i}: policy shape mismatch")
            for j in range(sizes[0]):
                ds.add_frame({
                    "observation.images.image": np.ascontiguousarray(f["image"][j, ::-1, ::-1]),
                    "observation.images.image2": np.ascontiguousarray(f["image2"][j, ::-1, ::-1]),
                    "observation.state": f["state"][j], "action": f["action"][j],
                    "task": "pick up the alphabet soup and place it in the basket",
                })
        ds.save_episode()
        parents.append({"demo_index": i, "frames": sizes[0],
                        "start_state_sha256": report["start_state_sha256"],
                        "report_sha256": digest(report_path), "frames_sha256": digest(frames_path)})
        print(json.dumps({"packed_demo": i, "frames": sizes[0]}), flush=True)
    ds.finalize()
    manifest = {
        "schema": "rg.public_h_native_dataset.v1", "source_repo": SOURCE_REPO,
        "revision": REVISION, "source_path": source_manifest["path"],
        "source_file_sha256": source_manifest["file_sha256_hf_api"],
        "license": "CC-BY-4.0", "attribution": "LIBERO original human demonstrations",
        "task": TASK, "selected_demo_indices": demos, "parents": parents,
        "fps_metadata": 20, "simulator_control_hz": 20, "control_alignment_verified": True,
        "raw_dt_seconds": .05, "pre_action_render": True,
        "camera_orientation_correction": "inverse 180-degree to raw P policy observations",
        "policy_camera_keys": ["observation.images.image", "observation.images.image2"],
        "state_dim": 8, "action_dim": 7, "human": True,
        "recovery_data": False, "browser_game_data": False,
        "held_out_native_start_ids": list(range(25, 50)),
    }
    (args.root / "RG_PUBLIC_H.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"dataset": str(args.root), "episodes": len(parents),
                      "frames": sum(x["frames"] for x in parents)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
