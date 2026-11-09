# WeRescue native GPU bridge

This service runs the actual pinned SmolVLA checkpoint and the actual LIBERO simulator. It has no scripted fallback. The UI may display a factory skin, but policy observations and all control/collection state come from the same native simulator. See [PROTOCOL.md](PROTOCOL.md).

## Runtime

Use the training-side pinned Linux runtime (`training/runtime/setup_pinned.sh` describes its exact LeRobot commit, torch and native dependencies). Do not install inference dependencies into the browser app or overwrite training code. Additional service packages:

```sh
python -m pip install -r inference/requirements.txt
```

Pinned LeRobot source: `b9cb121cb7d4e3e26ec5c906d08dda68d151ba52`. Pinned checkpoint: `lerobot/smolvla_libero` at `a24f1f380461b665158d52acd140152c79e8b36d`.

Start from the project root on the GPU host:

```sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export WERESCUE_DEVICE=cuda
# Optional: local snapshot directory with the exact pin, otherwise Hub downloads the exact revision.
# export WERESCUE_POLICY_PATH=/workspace/werescue-runtime/models/P
python -m inference.server
```

Default bind is `127.0.0.1:8000`. Forward that port with SSH to local port 8000; the UI's local Vite `/robot-api` proxy can use it. For an intentionally exposed GPU HTTP port, set `WERESCUE_HOST=0.0.0.0` **and** set `WERESCUE_API_TOKEN` to a secret; inject the bearer token from the UI's server-side proxy rather than shipping it in a browser bundle. CORS defaults to local Vite/preview origins. Set `WERESCUE_CORS_ORIGINS` to a comma-separated explicit list if needed.

Configuration:

- `WERESCUE_RECORD=1` (default) writes native PNG observations, state NPZ files, metadata and request/execution events under `inference/runs/<session-id>/`. Use `WERESCUE_RECORD_ROOT` for persistent GPU storage.
- API runtime default `WERESCUE_HORIZON=280` preserves the LIBERO-Object benchmark horizon. `run_server.sh` explicitly selects **1000** for interactive human collection; health/session metadata record this actual horizon. Benchmark evaluation remains a separate 280-step protocol.
- `WERESCUE_AUTOLOAD=0` is for API-only unit tests. It does not produce any policy actions.
- `WERESCUE_SUITE` and `WERESCUE_TASK_ID` default to `libero_object`, `0`.

## Verify actual readiness

`GET /health` is available during cold loading. `ready=true` means model/preprocessors loaded; it does not claim task success. Then reset the native session, fetch `/scene`, and step with `source:"policy"`. `inference_verified` only becomes true after actual model execution. A terminal `frame.success` comes from native LIBERO `check_success()`. One interactive episode does not establish a success rate.

Model failure remains an error/503 response. An incomplete or missing native environment remains a real error, never a replacement animation.

## Human collection

Take control through `/session/control` with `source:"operator"`, then submit bounded 7D relative OSC actions. Queued model actions are cleared on takeover/handback. Mark `recovery_complete` when annotating state 2; this retains operator control so subsequent actions are human task continuation R. Handing control back to policy creates a policy-generated suffix and must not be labeled human R.

Every stored observation precedes the corresponding native step input. `action_requested` and successful `executed_action` events are separate. The saved native state includes qpos/qvel, actuator/internal fields where available, and selected controller fields. Exact replay equivalence has not been demonstrated. A recovery boundary is an operator annotation, not an independent success predicate.

## Checks

```sh
python -m py_compile inference/server.py inference/runtime.py
WERESCUE_AUTOLOAD=0 python -m unittest inference.test_server
```

API tests use a clearly named fake runtime only inside the test module; they do not verify GPU inference, simulator rendering or model success. Actual inference validation must run on the GPU host.
