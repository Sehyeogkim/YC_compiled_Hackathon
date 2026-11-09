# Inference deployment

- Provider: RunPod, using a dedicated inference environment. Provision your own GPU host and SSH access.
- Verified hardware: NVIDIA A40.
- Image: `kis99156/wemine-robot:2026-10-23`.
- Connect through your own `GPU_HOST` SSH alias and forward port 8000 as described in the [setup guide](../docs/SETUP.md).
- Service root: `/workspace/werescue-ui-inference`.
- Isolated runtime: `/workspace/werescue-ui-inference/runtime/.venv`.
- API bind: `127.0.0.1:8000`, reached through SSH forwarding. The configured container port 8000 is not used as an unauthenticated public listener.
- Runtime installation log: `setup.log`; model service log: `server.log`; process ID: `server.pid`.
- Native recordings: `/workspace/werescue-ui-inference/runs`.

## Exact inputs

- LeRobot source revision: `b9cb121cb7d4e3e26ec5c906d08dda68d151ba52`.
- Policy: `lerobot/smolvla_libero` revision `a24f1f380461b665158d52acd140152c79e8b36d`, stored in `models/P` with `WERESCUE_PIN.json`.
- Native assets: `lerobot/libero-assets` revision `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`, stored in `assets` with `WERESCUE_ASSETS_PIN.json`.
- Native LIBERO-Object task 0, alphabet soup can into basket. Relative OSC, 20 Hz, eight state values, seven action values, two native cameras.
- Interactive collection horizon: 1000 steps, recorded in health/session metadata. Separate benchmark evaluation retains its own 280-step protocol.
- Native policy and native human controls share the authoritative simulator. The browser renders its exact state with `mj_forward` only.

## Current verification

Verified on 2026-11-09 at 02:50 UTC. Code compilation and eight API/contract tests pass locally. The exact model loaded on NVIDIA A40 and completed native reset plus 20 policy-driven simulator steps. `/health` reported `ready`, `model_ready`, `native_ready`, `native_step_verified`, and `inference_verified` true for that smoke session. The service ran as PID 2504 at the time of verification; inspect `server.pid` for its current PID. Restarting clears active sessions and verification flags until the next real reset/inference.

The inspectable result is [proof/native-policy-smoke.json](proof/native-policy-smoke.json). Fresh policy predictions took 632.43 ms on the first chunk and 256.49 ms on the second. The intervening nine actions per chunk reused the real policy queue and correctly report zero fresh inference time. Twenty steps advanced native simulation time from 0.5 to 1.5 seconds and changed the native end-effector position. This verifies inference and simulator integration; it does not establish task success or a benchmark success rate.

The native scene has `nq=58`, `nv=51`, and 82 referenced assets. The initial scene JSON embeds about 125 MB of base64 asset data. The controller reports 0.05 meters and 0.5 radians per normalized translation/rotation unit. The installed PandaGripper source confirms that action zero holds its current target. Frame and scene dimensions match.

Runtime packages: LeRobot 0.6.2 from the pinned source, torch 2.8.0+cu128, torchvision 0.23.0+cu128, torchcodec 0.7.0, transformers 5.5.4, hf-libero 0.1.4, robosuite 1.4.0, MuJoCo 3.3.7, NumPy 2.1.2. The required SmolVLM2 backbone/tokenizer were already available in `/opt/hf`; no replacement checkpoint was used. Resolved packages are recorded remotely in `runtime/resolved-requirements.txt`.

The observation preprocessing follows the pinned LeRobot evaluation implementation: `preprocess_observation`, attach the vector environment's `task_description`, then apply the environment processor. The removed `add_envs_task` helper is not used.

Training-fork compatibility note (training files were not modified): `training/runtime/native_smoke.py` currently imports `add_envs_task` from `lerobot.envs.utils`, but that helper does not exist in the exact pinned LeRobot revision. Replace that import with `preprocess_observation` only, and replace `env_pre(add_envs_task(vec, preprocess_observation(observation)))` with:

```python
batch = preprocess_observation(observation)
batch['task'] = list(vec.call('task_description'))
batch = env_pre(batch)
```

Apply the same sequence at its final-observation preprocessing call. This is the pinned `lerobot_eval.py` rollout path, verified by the actual inference smoke above.

`/session/mark` with `recovery_complete` keeps the human in control for subsequent human task-continuation data. No VLM is included in this service. Model policy queues are reset at control changes.

## Lossless speed settings, 04:08 UTC

A recorded configuration update preserved the paused episode and started a fresh episode with faster settings. That run used torch intra-op **4**, inter-op **1**, and PNG `compress_level=1`. PNG pixels, NPZ compression/arrays, model pin, native model/geometry, cameras, controller and 0.05-second simulation ticks remain unchanged. Health and new session metadata expose the CPU/storage configuration.

Original user session `1a178aa2-4962-425d-a917-3d91ea7d96a6` is archived at `/workspace/werescue-ui-inference/archives/speedup-20261109T040400Z`. Both the raw state before explicit reset and the raw state after saving its final observation are retained. A separate preservation note labels the interruption for configuration upgrade. The prior 1472 files are byte-identical, prior event bytes remain a prefix, and final step490 qpos/qvel exactly match the saved last frame. Final raw contains 982 camera PNGs and 491 state NPZs. See [preservation proof](proof/speedup-preservation.json).

After the update, an actual 40-step P test from initial0/seed123 used four ten-step HTTP requests from the local Mac through **Vite `/robot-api` and the same SSH WAN tunnel**. It advanced 2.0 native seconds in **6.073 client wall seconds** (6.59 observed Hz). The first 40 pre-update and post-update actions are exactly equal, as are both initial camera images. Comparing the same server-event interval for the same first40 actions gives **13.358 → 4.955 seconds**; mean native `env.step` time was **56.56 → 14.56 ms**. These are single-run integration measurements; request batching, CPU and storage settings changed together. They do not prove 20 Hz real-time execution or task success.

Proofs: [Vite/WAN client measurement](proof/policy-speed-after-vite-4threads-png1-batch10.json), [direct SSH tunnel client measurement](proof/policy-speed-after-4threads-png1-batch10.json), [equal-prefix comparison](proof/speedup-comparison.json). Performance test IDs and the intermediate no-action session are explicitly excluded in `test_sessions.json`. The verified API was loopback-only. These notes describe recorded runs, not current GPU availability.
