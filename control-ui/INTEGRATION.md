# Simulation and learning integration

## Current boundary

`RobotScene` receives `SceneCommand` requests and emits `SimSnapshot` observations. The React control desk records both using `createEpisodeRecorder()`. The default controller is a scripted browser demonstration; no external model has been connected.

The UI and simulation run locally in the browser. A future Python policy service can run on the same workstation or a remote GPU machine. That service does not require publishing the UI or deploying Cloudflare.

## Recorded episode v1

`src/protocol/types.ts` is authoritative. Export JSON has `schemaVersion: "werescue.episode.v1"`, one episode ID, UTC timestamps, monotonic elapsed time, and a sequence number for ordering. Each command links to its preceding observation sequence. A command is labeled `executionStatus: "requested"`; it must not be treated as a measured executed action.

- `snapshot`: reported joint/EEF/gripper/object state, simulation time, controller stage and task status.
- `command`: requested UI action and actor.
- `marker`: task, recovery, handoff and reset boundaries.
- `segment`: `task`, `recovery`, or `unassigned`.
- `controlSource`: `operator`, `demo_controller`, or `policy`.

`recovery_start` marks operator takeover. `recovery_end` and `policy_handoff` mark a request to resume the demo controller. A future backend must additionally record whether and when the handoff was accepted. A reset is explicitly marked and must not be counted as a successful recovery.

## Policy adapter still required

Do not feed `lastAction`, joint angles, or manual Cartesian deltas straight to a pretrained policy without establishing the control contract. Before connecting SmolVLA, agree on:

1. Observation RGB cameras, resolution, orientation, timestamps and preprocessing. Current exports include no RGB frames.
2. Robot state fields, units and coordinate frames. Confirm end-effector pose representation and gripper convention in the running simulator.
3. Policy action shape, translation and rotation convention, scaling, control rate, action chunking and the low-level controller.
4. Normalization statistics and task instruction expected by the exact checkpoint.
5. Requested versus accepted versus executed actions, latencies, watchdog and ownership handoff.
6. Episode stop, terminal success, recovery boundaries and separate held-out evaluation.

The custom conveyor scene has a domain gap relative to LIBERO. Report performance only after actual inference and evaluation. Keep scripted demonstration episodes distinguishable from model rollouts and operator demonstrations.

## Agreed collection flow (2026-11-09, UI chat)

VLM-based failure detection/explanation is excluded from this MVP. The operator watches the real VLA and decides when to take control. The Ghost UI appears after takeover and displays an operator-selected historical observation and recorded EEF path. A historical pose is guidance, not a verified safe state.

During collection the operator performs both segments: failed state 3′ to resumable state 2, then human task continuation 2 to completion 4. The recovery-complete marker changes only the segment label and keeps operator control. The primary UI does not hand control back to the VLA at the boundary. Reset/next episode happens after the episode ends and restores native robot/object initial state and policy queues; it is not part of the recovery segment.

The GPU's native LIBERO simulator and original camera observations remain authoritative. The Three.js control scene is a visualization of the same native qpos/qvel, loaded from the same MJCF/assets. It calls mj_forward only. Offline simulation is explicitly separate and cannot be presented as actual VLA execution.
