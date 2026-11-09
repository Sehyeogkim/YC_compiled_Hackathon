#!/usr/bin/env python3
"""P→R pilot on native-verified human Physical Arena task episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from download_checkpoint import MODEL_ID, MODEL_REVISION

TASK = "libero_object__pick_up_the_alphabet_soup_and_place_it_in_the_basket"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--P", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--require-camera-correction", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch size must be positive")
    if args.output.exists():
        parser.error("output path already exists")
    pin_file = args.P / "WERESCUE_PIN.json"
    if not pin_file.is_file() or json.loads(pin_file.read_text()) != {"repo_id": MODEL_ID, "revision": MODEL_REVISION}:
        parser.error("P checkpoint pin mismatch")
    config_file = args.P / "config.json"
    if not config_file.is_file():
        parser.error("missing P config")
    config = json.loads(config_file.read_text())
    if (config.get("input_features", {}).get("observation.state", {}).get("shape") != [8]
            or config.get("output_features", {}).get("action", {}).get("shape") != [7]
            or not all(key in config.get("input_features", {})
                       for key in ("observation.images.camera1", "observation.images.camera2"))):
        parser.error("P state/action/two-camera contract mismatch")
    info_file = args.dataset_root / "meta/info.json"
    provenance_file = args.dataset_root / "RG_ARENA_NATIVE_REPLAY.json"
    if not info_file.is_file() or not provenance_file.is_file():
        parser.error("dataset needs LeRobot meta/info.json and RG_ARENA_NATIVE_REPLAY.json")
    info = json.loads(info_file.read_text())
    prov = json.loads(provenance_file.read_text())
    parents = prov.get("parent_episodes")
    if (prov.get("schema") != "rg.arena_native_task_dataset.v1"
            or prov.get("task_id") != TASK
            or prov.get("source") != "human_browser_unverified"
            or prov.get("control_source") != "human_game_play"
            or prov.get("native_replay_verified") is not True
            or prov.get("training_eligible") is not True
            or prov.get("recovery_data") is not False
            or prov.get("fps") != 20 or prov.get("state_dim") != 8 or prov.get("action_dim") != 7
            or not isinstance(parents, list) or not parents
            or len({p.get("episode_id") for p in parents}) != len(parents)
            or prov.get("eval_parent_ids") != []):
        parser.error("invalid native human R provenance")
    if args.require_camera_correction and (
            "inverse 180-degree" not in prov.get("camera_orientation_correction", "")
            or prov.get("task_text_correction", {}).get("to") !=
            "pick up the alphabet soup and place it in the basket"):
        parser.error("corrected P camera orientation and exact task text are required")
    source = Path(prov["source_bundle"])
    if source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() != prov.get("source_bundle_sha256"):
        parser.error("source browser bundle hash mismatch")
    if info.get("fps") != 20 or info.get("total_episodes") != len(parents):
        parser.error("LeRobot episode/FPS mismatch")
    features = info.get("features", {})
    if features.get("observation.state", {}).get("shape") != [8] or features.get("action", {}).get("shape") != [7]:
        parser.error("LeRobot state/action mismatch")
    for key, policy_key in (("observation.images.image", "observation.images.camera1"),
                            ("observation.images.image2", "observation.images.camera2")):
        expected = config["input_features"][policy_key]["shape"]
        if features.get(key, {}).get("shape") != [expected[1], expected[2], 3]:
            parser.error(f"LeRobot camera mismatch: {key}")
    cmd = [sys.executable, str(Path(__file__).with_name("train_preserve_stats.py")),
           f"--policy.path={args.P.resolve()}", f"--dataset.repo_id={args.dataset_id}",
           f"--dataset.root={args.dataset_root.resolve()}", f"--output_dir={args.output.resolve()}",
           "--job_name=rg_arena_P_plus_R_pilot", f"--steps={args.steps}",
           f"--batch_size={args.batch_size}", f"--seed={args.seed}",
           "--num_workers=0", "--dataset.video_backend=pyav", "--policy.device=cuda",
           "--policy.n_action_steps=10", "--policy.freeze_vision_encoder=true",
           "--policy.train_expert_only=true", "--policy.train_state_proj=true",
           "--policy.optimizer_lr=0.00003", f"--policy.scheduler_warmup_steps={min(10,args.steps//10)}",
           f"--policy.scheduler_decay_steps={args.steps}", "--policy.scheduler_decay_lr=0.000003",
           "--policy.push_to_hub=false", "--wandb.enable=false", f"--save_freq={args.steps}",
           "--log_freq=1", '--rename_map={"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}']
    print(json.dumps({"kind": "P_plus_R_human_arena_native_pilot", "P": str(args.P),
                      "dataset": str(args.dataset_root), "parents": len(parents),
                      "normalization": "retain_P_statistics", "steps": args.steps,
                      "execute": args.execute, "command": cmd}, indent=2), flush=True)
    print(shlex.join(cmd), flush=True)
    if not args.execute:
        return
    subprocess.run(cmd, check=True)
    checkpoint = args.output / "checkpoints" / f"{args.steps:06d}" / "pretrained_model"
    if not (checkpoint / "model.safetensors").is_file():
        raise RuntimeError(f"missing trained checkpoint: {checkpoint}")
    record = {"schema_version": "werescue.human_finetune.v1", "arm": "task",
              "base_policy": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
              "dataset": str(args.dataset_root),
              "parent_episode_ids": [p["episode_id"] for p in parents], "checkpoint": str(checkpoint),
              "normalization": "retain_P_statistics", "steps": args.steps,
              "human_dataset_provenance": prov,
              "training_source": "native_replayed_human_game_task_only", "recovery_claim": False,
              "qualification_scope": "P+human game task R pilot; no policy failure or recovery claim"}
    payload = json.dumps(record, indent=2) + "\n"
    (args.output / "RG_ARENA_P_PLUS_R.json").write_text(payload)
    (checkpoint / "RG_ARENA_P_PLUS_R.json").write_text(payload)
    (checkpoint / "WERESCUE_HUMAN_FINETUNE.json").write_text(payload)


if __name__ == "__main__":
    main()
