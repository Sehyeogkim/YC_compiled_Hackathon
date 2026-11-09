# Native collection → training handoff

The live service is on the **inference** A40 Pod `cg1lqt1bke465j`, not the training Pod. Do not restart it during collection. Native raw sessions are under `/workspace/werescue-ui-inference/runs/<session-id>/`. The service never uses independent browser physics as training observations.

## Collect now

Use initial-state IDs **0–19** for training collection. IDs 20–24 remain validation, 25–44 task evaluation, 45–49 video. Monitor P, take control when it fails, perform human recovery R′, mark recovery complete, **retain human control**, then complete the task R until native LIBERO SUCCESS. Reset afterward. A marker is the operator's annotation; it does not claim an automatic/independent recovery check. Keep incomplete/aborted attempts in raw storage.

Before production collection, reset once to get a new session ID. Integration/smoke/performance session IDs in [test_sessions.json](test_sessions.json) are always excluded by the exporter, including `ea24c94f-8595-4285-ae4e-907044e42561`, `50d16d9d-b4c1-485e-98d6-96b48d605fff` and `cc95ee9f-423d-4dda-8a7f-2ddbd08f6ae1`. Do not reuse a test session for training collection. Operator identity is supplied explicitly during export.

## One command after collection

On the inference Pod, after the finished sessions are safely written:

```bash
cd /workspace/werescue-ui-inference
runtime/.venv/bin/python export_segments.py runs \
  --output exports/human-collection-01 --operator-id jeff \
  --accept-operator-annotations
```

The explicit annotation option matches today's workflow. It selects only native-successful, contiguous human R′+R pairs with one marker; labels them `qualification=operator_annotated`, **`recovery_predicate_verified=false`**. Without this option, recovery annotations remain unqualified candidates unless a separate verified evidence JSON is supplied with `--recovery-evidence-root`. It never invents an independent predicate or task success.

`exports/human-collection-01/manifest.json` contains `training_selection`: the eligible sessions only. Each eligible session contains `R_prime.npz`, `R.npz`, and `export.json`. Ineligible but alignable pairs go under `<id>/candidate/`; incomplete observations are rejected with an explicit reason. Raw PNG/state/event files are never rewritten/deleted. The exported manifest is **not** the stricter `werescue.collection.v1` validator input and does not assert LeRobot readiness.

Download both selected raw session directories and export files. PNG references in NPZ are relative to the session's `raw_root`, which must be remapped to its downloaded location. An entire raw snapshot is easiest for this short collection:

```bash
mkdir -p inference/collected/raw inference/collected/exports
scp -i /Users/jeff/.ssh/id_ed25519_runpod -P 22075 -r \
  root@69.30.85.73:/workspace/werescue-ui-inference/runs/. inference/collected/raw/
scp -i /Users/jeff/.ssh/id_ed25519_runpod -P 22075 -r \
  root@69.30.85.73:/workspace/werescue-ui-inference/exports/human-collection-01 \
  inference/collected/exports/
```

Take a snapshot after terminal success or explicit reset, rather than copying a session while it is actively writing. Downloading all raw tests does not make them training-eligible; `training_selection` controls selection. `inference/collected/` and exports are ignored by Git.

## Exact native alignment

Raw session files:

- `metadata.json`: session/model/framework/assets pins, task, initial state, seed, 20 Hz, horizon1000, scene hash.
- `events.jsonl`: actual `executed_action` after native `vec.step` returns; `action_requested` is separate and is **not** treated as execution. Policy and operator source plus recovery/task segment are on every event.
- `000123.state.npz`: 8D **pre-normalization** `observation_state`, flattened native simulator integration state, qpos/qvel, actuator ctrl, warm-start state, mocaps, controller goals/orientation, simulation time.
- `000123.agentview.png`, `000123.wrist.png`: lossless native camera RGB, 256×256, already flipped on height and width to match the pinned policy's input orientation. **Do not flip these images again.** Live JPEG previews and browser imagery are not training sources.

For `executed_action` at native observation step **t**, use state/images **t** → action7 **t** → state/images **t+1**. Each control tick advances 0.05 native simulation seconds, regardless of slower wall time. The final native observation is saved at terminal completion or before an explicit reset. A paused, nonterminal last frame may not yet be on disk; exporter refuses to synthesize it.

At marker step **b**, R′ includes actions `[takeover_step, b)` and R includes `[b, terminal_step)`. Thus R′'s final next observation is exactly R's first observation. Policy prefix is excluded. Policy actions inside the human intervention, multiple markers, missing observations, resets before success, reserved initial-state IDs, missing operator ID, and test sessions are excluded from default training selection.

## Export schema and training-side differences

The NPZ uses collector-compatible core keys `observation_state[N,8]`, `action[N,7]`, `next_observation_state[N,8]`, `sim_state_before/after`, `actuator_ctrl_after`, `frame_index`, `phase` (0 recovery, 1 task), `task_success`, `sim_timestamp`, `next_sim_timestamp`. Additional `native_step_index`, `png_refs[N,2]`, `next_png_refs[N,2]`, and `control_source` preserve lineage. Arrays contain no pickled objects.

Differences from `training/collector/server.py`:

- Its PNGs are native unflipped pixels in `agentview/` and `wrist/`; ours are **already policy-oriented**, named by native step. Convert only once.
- Its recovery boundary passes a native resumable-state predicate. Today's UI boundary is a human annotation, explicitly accepted by the export flag; downstream reports must retain that distinction.
- Its wall timestamps are monotonic capture/dispatch times. Our historical service records UTC action request/execution events. Exported `timestamp`/`action_dispatch_timestamp` are request UTC minus session creation UTC, **not capture timestamps**; `next_timestamp` is native-step completion UTC. Use actual `sim_timestamp`/`next_sim_timestamp` for the 20 Hz dataset grid. Observation capture wall timestamps are unavailable and are not fabricated.
- Ours saves qpos/qvel and controller goals as well as flattened state. Exact native replay/restoration is still not certified by mere serialization.
- Our NPZ + PNG references require a LeRobot dataset conversion; `training/runtime/launch_finetune.py` intentionally rejects raw NPZ without `meta/info.json`.

Conversion owner: training fork. Read `manifest.json.training_selection`, resolve PNGs against downloaded raw UUID directories, create separate R and R′ datasets at **20 Hz**, preserve 8D state and 7D native relative OSC actions. Expose cameras as `observation.images.image` and `image2` (the training launcher renames them to checkpoint `camera1`, `camera2`). Preserve P's processor statistics and initialize both arms independently from exact P; do not initialize one finetune from the other. Keep parent session/initial-state lineage together across splits.

Native task success is proven by `episode_terminal.success=true` with predicate `native_LIBERO_check_success`. That is not a held-out success rate or proof of learned recovery. No training result is claimed by exporting raw human demonstrations.
