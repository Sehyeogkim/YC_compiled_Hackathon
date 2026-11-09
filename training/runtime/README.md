# RG Service Center · Native LIBERO runtime

Default task is **LIBERO-Object task 0: pick up the alphabet soup and place it in the basket**. This is a native tabletop scene with the soup can, distractors and basket. It is not a conveyor simulation. Collection, learning and evaluation must use this same native environment.

## Fast path: existing RunPod environment first

Do not run the optional clean setup until the prebuilt environment has been checked. Activate its existing Python environment and run:

```sh
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python training/runtime/native_smoke.py \
  --policy /workspace/werescue-ui-inference/models/P --suite libero_object --task-id 0 \
  --init-state 0 --seed 123 --steps 300 --n-action-steps 10 \
  --device cuda --out /workspace/werescue/runs/P-smoke-001
```

The existing RunPod checkpoint is `/workspace/werescue-ui-inference/models/P`, pinned to `lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d`. `download_checkpoint.py` can recreate that pin on a fresh machine. Use P's own camera1/camera2 processor and normalization statistics. The execution override NAS10 matches the Control UI and baseline pilots; the checkpoint config default is NAS50.

`native_smoke.py` uses the installed environment's own LeRobot processor stack and records actual postprocessed actions passed to the native simulator. Its CLI has not been rerun after the newer LeRobot API change; use the verified `training/evaluation/run_hfvla50.py` runtime for current native results. The older smoke script saves:

- `run.json`, `summary.json`: package versions, task/start/seed/control configuration and measured outcome.
- `agentview.mp4`, `wrist.mp4`, `side_by_side.mp4`: actual two native RGB cameras, flipped the same way as the LeRobot processor.
- `trajectory.npz`: state8 observations, executed action7, elapsed timestamps. There are N actions and N+1 observations.
- `native_states.npz`: per-observation flattened integration state, qpos/qvel/ctrl and available actuator/mocap/warmstart/controller fields.
- `scene.xml`: model provenance. Full snapshot restore equivalence still needs testing; qpos alone is insufficient.

The script never injects scripted recovery or fabricates success. A timed-out rollout is a valid smoke result and remains a failure. It is one rollout, not a benchmark estimate.

## Optional isolated framework setup

If the prebuilt runtime cannot run the smoke, use a fresh directory:

```sh
bash training/runtime/setup_pinned.sh /workspace/werescue-runtime
source /workspace/werescue-runtime/env.sh
```

Requires Linux, `uv`, git, NVIDIA driver access, EGL/GL libraries and network. It creates Python 3.12 and a clean LeRobot checkout pinned to `b9cb121cb7d4e3e26ec5c906d08dda68d151ba52`, with pinned `hf-libero`, torch/torchvision/torchcodec and MuJoCo. Remaining resolved packages are saved in `resolved-requirements.txt`; this is not a fully locked transitive install until that resolution has been recorded. It does not alter the existing prebuilt environment or local project checkout. No apt changes, cloud provisioning or GPU training are performed by these files automatically.

## Human collection architecture

Run **one native environment process on the GPU machine**, not a second browser physics world. The Control UI receives encoded native agentview/wrist frames plus native state and sends operator actions to that process. Browser 3D visualizes the received state. There must be one authoritative simulator.

1. P performs the task and the server retains snapshots and observation/action history.
2. On a demonstrated failure, stop policy action dispatch and mark `recovery_start` at the observed failed state. Any injected failure must be separately labeled; it is not an observed model failure.
3. The operator's controls produce bounded normalized **native OSC pose-delta6+gripper** actions. Execute and record at the environment's 20 Hz control rate. Keyboard requests and executed actions are distinct records. A stale input watchdog stops dispatch rather than holding the last move indefinitely.
4. The operator recovers `3′→2` and marks **Recovery complete → task**. The implemented boundary is an operator annotation, not an independently verified recovery predicate. An independent resumability check must inspect object/contact/gripper state as well as arm pose; a candidate target is an open-gripper pre-grasp hover above an available can. Exact task tolerances and physical safety are not established by the annotation.
5. Keep the operator in control to complete `2→3→4`; otherwise the suffix is policy-generated and is not human R. Mark task completion using native LIBERO `check_success()`.
6. Preserve the original episode, then cut two **separate LeRobot episodes**: `R_prime=[recovery_start,handoff)`, `R=[handoff,task_end)`. Prevent 50-step action chunks from crossing this boundary. Retain episode/initial-state lineage and raw unsuccessful attempts separately.
7. Assign train/val/test by parent episode and initial-state lineage before frame sampling. Do not evaluate on a failure snapshot derived from a training episode and label it unseen.

This directory provides the native runtime setup, raw-to-LeRobot converter and training launchers. Live transport and native human recording are implemented in [`../../control-ui/src/native/`](../../control-ui/src/native/) and [`../../inference/`](../../inference/). [`../../inference/export_segments.py`](../../inference/export_segments.py) exports eligible native human segments; [`convert_human_segments.py`](convert_human_segments.py) converts them into separate R′ and R datasets with provenance checks.

The tooling is implemented, while qualified human datasets, P+R/P+R′ training checkpoints and measured intervention-improvement results remain pending. Native policy smoke exports and the separate practice dataset are not human demonstrations.

## Independent finetuning branches

### Separate HFVLA training-pipeline rehearsal (practice data only)

`build_practice_dataset.py` converts **one completed policy rollout** with
`done.json`, `trace.jsonl.gz`, and `rollout.mp4` into a one-episode LeRobot
dataset. The evaluator saved the two cameras side by side and rotated each by
180 degrees; the converter reverses that rotation and aligns each pre-action
RGB/state observation with the executed native action at 20 Hz. It writes
`WERESCUE_PRACTICE.json` with source hashes and the policy checkpoint ID.

```sh
python training/runtime/build_practice_dataset.py \
  --trial /workspace/werescue-evaluation/runs/hfvla-can-dev50/task-00_init-01_rng-1000 \
  --output /workspace/werescue-evaluation/practice/hfvla-success-one \
  --repo-id werescue/practice_hfvla_self_imitation

python training/runtime/launch_practice_finetune.py \
  --P /workspace/werescue-evaluation/models/HFVLA \
  --dataset-root /workspace/werescue-evaluation/practice/hfvla-success-one \
  --dataset-id werescue/practice_hfvla_self_imitation \
  --output /workspace/werescue-evaluation/practice/hfvla-train-20 \
  --steps 20 --batch-size 2 --execute
```

This separate practice launcher accepts only the pinned `HuggingFaceVLA/smolvla_libero`
checkpoint, preserves its normalization statistics, trains the action expert
and state projection, and saves the final checkpoint. It defaults to a dry run;
`--execute` performs training. This **P self-imitation smoke** checks the
data→training→checkpoint path. It is neither human R nor R′ and cannot support
the P versus P+R graph or recovery claim. Use a separate held-out native start
for a one-trial checkpoint load/evaluation smoke; its result is not a success
rate estimate. The actual R and R′ branches below require human intervention
data and independent training from the same P.

After collecting up to 10 human interventions, export eligible UI sessions and
convert the two segments. `--accept-operator-annotations` keeps unverified
recovery boundaries explicitly marked as annotations; omit it when independent
recovery evidence is available. The converter accepts 1–10 qualified source
sessions, writes both LeRobot datasets and their provenance sidecars, and
rejects policy actions, reset segments, missing frames, or failed task endings.

```sh
/workspace/werescue-ui-inference/runtime/.venv/bin/python \
  /workspace/werescue-ui-inference/inference/export_segments.py /workspace/werescue-ui-inference/runs \
  --output /workspace/werescue-evaluation/human_exports \
  --operator-id jeff --accept-operator-annotations

/workspace/werescue-ui-inference/runtime/.venv/bin/python \
  /workspace/werescue-evaluation/training/runtime/convert_human_segments.py \
  --manifest /workspace/werescue-evaluation/human_exports/manifest.json \
  --raw-root /workspace/werescue-ui-inference/runs \
  --exports-root /workspace/werescue-evaluation/human_exports \
  --output-root /workspace/werescue-evaluation/human
```

The converter restores the native policy-view orientation from the Control
UI's upright 256×256 PNGs. It creates `WERESCUE_SEGMENTS.json` and
`WERESCUE_HUMAN_DATASET.json` in each dataset root. The launcher checks those
sidecars, the state8/action7/image+image2/20Hz schema and the source policy.
The current UI and canonical P both use `lerobot/smolvla_libero` with NAS10.
Source file checks do not prove exact simulator replay or independent recovery
verification. Practice data is rejected from the human training arms.

```sh
python training/runtime/launch_finetune.py \
  --arm task --P /workspace/werescue-ui-inference/models/P \
  --dataset-root /workspace/werescue-evaluation/human/R --dataset-id werescue/human_native_R \
  --output /workspace/werescue-evaluation/human_runs/P-plus-R

python training/runtime/launch_finetune.py \
  --arm recovery --P /workspace/werescue-ui-inference/models/P \
  --dataset-root /workspace/werescue-evaluation/human/R_prime --dataset-id werescue/human_native_R_prime \
  --output /workspace/werescue-evaluation/human_runs/P-plus-R-prime
```

These commands validate the canonical `lerobot/smolvla_libero` checkpoint pin, human segment provenance, and dataset schema before printing the exact command. `image/image2` dataset keys are renamed to the P checkpoint's `camera1/camera2`, using P's own normalization statistics. Add `--execute` to actually train. Both start from **P**, independently. Time-budgeted first-pass settings with up to 10 qualified interventions: expert-only SFT, frozen vision, trainable state projection, batch8, learning rate3e-5, 300 steps, warmup30, seed1000, checkpoint at step300. These are starting settings, not validated improvement claims; actual run time and loss must be measured on R and R′. No simulator environment is configured in the training launcher, so simulator evaluation runs separately.

The wrapper retains P normalization statistics and checks the installed SmolVLA `forward` source. It only adds an `action_is_pad→actions_id_pad` alias when the old reader is actually detected; the newer reader is left untouched. The framework source is not edited. The first run still needs an actual batch/loss/padding smoke before either long finetune. `launch_finetune.py` checks tensor schema metadata, not split leakage, action semantics or demonstration quality.

Task comparison uses the same held-out native initial states, seed, action-execution horizon, control mode and native success predicate for P and P+R. Recovery video starts from a held-out failed-state snapshot and ends once the recovery predicate passes. No reset is allowed within the recovery rollout.

## Provenance

The smoke API pattern follows Jeff's earlier WEMINE helper `game1016/lerobot_export/libero_failure_eval.py`. It calls installed Hugging Face LeRobot and LIBERO code without vendoring their implementations. Framework and checkpoint licenses are Apache-2.0; preserve their source licenses when distributing a runtime.
