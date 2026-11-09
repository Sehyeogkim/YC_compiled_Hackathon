#!/usr/bin/env python3
"""Fine-tune pinned P on verified public LIBERO human demonstrations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

MODEL_ID = "lerobot/smolvla_libero"
MODEL_REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"
SOURCE_REPO = "yifengzhu-hf/LIBERO-datasets"
REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--P", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        p.error("output exists")
    pin = json.loads((args.P / "WERESCUE_PIN.json").read_text())
    if pin != {"repo_id": MODEL_ID, "revision": MODEL_REVISION}:
        p.error("P pin mismatch")
    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    prov = json.loads((args.dataset_root / "RG_PUBLIC_H.json").read_text())
    demos = prov.get("selected_demo_indices")
    if (prov.get("schema") != "rg.public_h_native_dataset.v1"
            or prov.get("source_repo") != SOURCE_REPO or prov.get("revision") != REVISION
            or not prov.get("human") or prov.get("browser_game_data") or prov.get("recovery_data")
            or not prov.get("control_alignment_verified")
            or prov.get("fps_metadata") != 20 or prov.get("simulator_control_hz") != 20
            or prov.get("state_dim") != 8 or prov.get("action_dim") != 7
            or "inverse 180-degree" not in prov.get("camera_orientation_correction", "")
            or not isinstance(demos, list) or not demos or any(not isinstance(i, int) or i < 0 or i >= 25 for i in demos)
            or info.get("fps") != 20 or info.get("total_episodes") != len(demos)):
        p.error("public H dataset provenance/schema mismatch")
    cmd = [sys.executable, str(Path(__file__).with_name("train_preserve_stats.py")),
           f"--policy.path={args.P.resolve()}", f"--dataset.repo_id={args.dataset_id}",
           f"--dataset.root={args.dataset_root.resolve()}", f"--output_dir={args.output.resolve()}",
           "--job_name=rg_P_plus_public_H_pilot", f"--steps={args.steps}",
           f"--batch_size={args.batch_size}", "--seed=1000", "--num_workers=0",
           "--dataset.video_backend=pyav", "--policy.device=cuda",
           "--policy.n_action_steps=10", "--policy.freeze_vision_encoder=true",
           "--policy.train_expert_only=true", "--policy.train_state_proj=true",
           "--policy.optimizer_lr=0.00003",
           f"--policy.scheduler_warmup_steps={min(10, args.steps//10)}",
           f"--policy.scheduler_decay_steps={args.steps}", "--policy.scheduler_decay_lr=0.000003",
           "--policy.push_to_hub=false", "--wandb.enable=false", f"--save_freq={args.steps}",
           "--log_freq=1",
           '--rename_map={"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}']
    print(json.dumps({"dataset": str(args.dataset_root), "public_demos": demos, "steps": args.steps,
                      "execute": args.execute}), flush=True)
    if not args.execute:
        return
    subprocess.run(cmd, check=True)
    checkpoint = args.output / "checkpoints" / f"{args.steps:06d}" / "pretrained_model"
    if not (checkpoint / "model.safetensors").is_file():
        raise RuntimeError("checkpoint missing after train")
    record = {
        "schema_version": "werescue.human_finetune.v1", "arm": "task",
        "base_policy": {"repo_id": MODEL_ID, "revision": MODEL_REVISION},
        "training_source": "public_LIBERO_human_teleoperation", "recovery_claim": False,
        "dataset": str(args.dataset_root), "human_dataset_provenance": prov,
        "steps": args.steps, "normalization": "retain_P_statistics",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest(),
    }
    payload = json.dumps(record, indent=2) + "\n"
    (args.output / "RG_P_PLUS_PUBLIC_H.json").write_text(payload)
    (checkpoint / "WERESCUE_HUMAN_FINETUNE.json").write_text(payload)
    print(json.dumps({"checkpoint": str(checkpoint), "public_demos": len(demos)}), flush=True)


if __name__ == "__main__":
    main()
