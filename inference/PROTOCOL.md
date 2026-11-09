# WeRescue native inference API v1

One serialized GPU/native simulator session. Bind to `127.0.0.1:8000` and reach it with an SSH tunnel. Optional `WERESCUE_API_TOKEN` requires `Authorization: Bearer ...` on every request. CORS allows localhost UI origins only by default. No scripted fallback exists.

All session frame envelopes (including reset, step, control, and mark) include top-level `scene_sha256`. Cache the exact `/scene` model/assets by this hash and reuse them across resets when it matches. Large responses negotiate gzip through `Accept-Encoding: gzip`; clients should allow their HTTP stack to decompress normally.

Pinned policy: `lerobot/smolvla_libero` @ `a24f1f380461b665158d52acd140152c79e8b36d`; native environment: `libero_object`, task 0, soup can into basket, relative OSC at 20 Hz. Runtime verifies task language and dimensions. `n_action_steps=10` uses ten actions from each 50-step policy prediction before refreshing observations, matching `training/experiment.json`.

## Calls

- `GET /health` → readiness, GPU, checkpoint, actual model-load error. HTTP service existence is not model readiness.
- `POST /session/reset` body `{ "initial_state_id": 0, "seed": 123, "control_source": "policy" }` → `{session_id, frame}`. This replaces the previous session and explicitly resets policy queues and native state.
- `GET /session/frame?session_id=...` → `{session_id, frame}`. Reads authoritative native cameras/telemetry without advancing physics.
- `GET /scene?session_id=...` → `{xml,assets:[{path,data,mime}],nq,nv,body_names,geom_names,site_names,cameras,coordinate_frame,scene_sha256}`. Assets are base64. Mount XML as `scene.xml` and each asset at its exact `path`. Only files referenced by the trusted native MJCF are bundled. Load this model once in browser MuJoCo, copy native `frame.qpos`/`frame.qvel`, call `mj_forward` only, and render its geoms. Do not step browser physics or run a separate browser controller in native mode.
- `POST /session/control` body `{session_id, source:"operator"|"policy"}` → `{session_id,frame}`. Takeover and handback reset policy queues. Switching control does not itself move the robot.
- `POST /session/step` body `{session_id,source:"policy",steps:1}` executes actual pretrained inference. `steps` is 1–10; the response contains `{session_id,frame,executed_actions,inference_ms}`. Stop polling when `frame.done` is true. The client owns wall-clock scheduling; simulation control steps are 0.05 s.
- Policy actions use the checkpoint's ten-action execution queue. `fresh_predictions` counts actual neural predictions during this request; `model_inference` is true only when this count is positive. `inference_ms` measures fresh prediction calls; `policy_call_ms` also includes cached action selection. A cached action remains policy-generated, but is not another neural inference.
- Operator stepping uses `{session_id,source:"operator",action:[dx,dy,dz,rx,ry,rz,grip],steps:1}`. Actions are **7D native normalized relative OSC**, each component in `[-1,1]`; `grip=-1` opens, `+1` closes. They are not browser joint targets or direct meter/radian deltas. Server validates shape, finite values and bounds; it never silently truncates 8D browser controls.
- `POST /session/mark` body `{session_id,marker:"recovery_complete",detail:"..."}` marks the human recovery/task boundary. **It keeps the operator in control**, for human task-continuation R collection. This is an annotation, not independently verified recovery success. `marker:"note"` also exists.
- `GET /session/export?session_id=...` → the session's metadata and executed-action/marker history. Native camera pixels and full integration-state arrays are also saved in the session directory on the GPU host when recording is enabled.
- `POST /predict` body `{camera1:"<base64 RGB JPEG/PNG>",camera2:"...",state:[8 floats],task:"...",reset:true}` → `{actions:[[7 floats]...],action_dim:7,chunk_size:50,execute_steps:10,inference_ms,model}`. Inputs must already be upright RGB, exactly 256×256, matching native frame orientation. This is actual pretrained inference. Standalone prediction is refused while a native session is active, keeping its queued actions isolated.

## Frame

`/scene` (alias `/session/scene`) also returns `control`: `translation_scale_m:[3]`, `rotation_scale_rad:[3]`, corresponding offsets, raw `input_min/max` and `output_min/max`, `gripper_open:-1`, `gripper_close:1`, `control_hz:20`, `dt_seconds:0.05`, `action_order:[dx,dy,dz,rx,ry,rz,gripper]`, `position_frame:"world"`. Scales are computed from the actual native OSC controller. Convert desired meter jogs using these reported scales/offsets, then respect normalized input limits. Do not copy the browser controller's 8D joint targets.

Health exposes both `ready` and `model_ready` booleans. `environment.native_ready` becomes true after the first native reset. `inference_verified` becomes true only after an actual model inference completes.

```json
{
  "cameras": {
    "agentview": {"mime":"image/jpeg","data":"BASE64","width":256,"height":256},
    "wrist": {"mime":"image/jpeg","data":"BASE64","width":256,"height":256}
  },
  "state": [8],
  "joints": [7],
  "eef": [3],
  "gripper": [2],
  "qpos": [],
  "qvel": [],
  "eef_orientation": [9],
  "body_poses": {"names":[],"xpos":[],"xquat":[]},
  "sim_time": 0.5,
  "step": 0,
  "success": false,
  "done": false,
  "control_source": "policy",
  "segment": "task",
  "task": "pick up the alphabet soup and place it in the basket"
}
```

Arrays `[8]`, `[7]`, etc. illustrate lengths, not values. `state` is the actual 8D pre-normalization policy proprioception. `eef` and gripper displacement are meters in native Z-up world coordinates, joints are radians. Native image orientation follows LeRobot's LIBERO processor (flip image H and W). `success` comes only from LIBERO's native `check_success()`.

The custom Three.js factory simulation remains a separate environment. A policy connected to native LIBERO must display these native camera frames, not independent browser geometry as if it were authoritative. The native cameras and native step state come from the same simulator. No scores or training results are fabricated.
