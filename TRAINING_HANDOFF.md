# WeRescue training handoff

## Ownership and integration boundary

The control UI is under `control-ui/`. Its React interface, browser simulation, and protocol are being developed together. Training work should run independently and avoid editing UI, simulation, protocol, package manifests, or lockfiles. Put training scripts, checkpoints, evaluation outputs, and dataset conversions in a separate directory or isolated checkout. Coordinate any contract change with the UI owner before implementation.

The initial customer is a robot supplier remotely managing deployed fleets across factories and logistics centers. Immediate value is understanding an incident, recovering the task, and resuming operation. Cross-site recovery data may support broader model improvement; generalization and cross-customer training rights remain unverified.

The authoritative simulation scene is [`control-ui/public/models/panda/factory.xml`](control-ui/public/models/panda/factory.xml), which the browser loads directly with local assets. Python training can load the same scene with `mujoco.MjModel.from_xml_path(...)` instead of recreating the robot or workcell. Keep relative asset paths intact. The conveyor feed uses a velocity/orientation servo approximation before contact-based grasping, so it does not establish accurate belt-contact physics. See [`control-ui/THIRD_PARTY_NOTICES.md`](control-ui/THIRD_PARTY_NOTICES.md) for asset provenance and licensing.

## Stable UI boundary

Canonical source: [`control-ui/src/protocol/types.ts`](control-ui/src/protocol/types.ts).

```ts
type SceneCommand = {
  id: number;
  type: 'start' | 'pause' | 'reset' | 'inject_failure' | 'take_control'
    | 'resume_policy' | 'stop' | 'move' | 'gripper';
  delta?: [number, number, number];
  gripper?: number;
};
```

Each command ID is consumed once. Commands express requests; an external controller must separately report whether an action was accepted, executed, limited, or rejected. The scene owns safety constraints and actuator execution.

`SimSnapshot` contains `status`, `simTime`, `joints`, `eef`, `gripper`, `canPosition`, `targetPosition`, `retries`, `taskProgress`, `beltRunning`, `message`, `physicsReady`, `lastAction`, and `success`, plus optional `stage` and `contactCount`. Use the exported TypeScript type instead of independently recreating it.

Confirmed scene conventions:

- `simTime`: MuJoCo simulation seconds; event elapsed time remains monotonic milliseconds.
- `joints`: seven arm joint positions in radians.
- `eef`, `canPosition`, `targetPosition`, and move `delta`: meters in MuJoCo's Z-up world frame. Robot base is `[0, 0, 0]`; belt motion is along Y. TCP is the hand site offset by 0.1034 m. Bin center is approximately `[0.32, -0.40, 0.055]`.
- `gripper` command: normalized `0` closed, `1` open. Snapshot gripper is finger displacement divided by 0.04 m.
- `lastAction`: current applied **8D `data.ctrl`**, ordered as seven arm joint targets and gripper control `[0, 255]`. This is not the Cartesian jog request or measured achieved motion. Error snapshots can have an empty vector.
- Control and recording frequency must be established by the model bridge; browser render intervals do not define a fixed training sample rate.

The current native success predicate runs after the placement sequence: can-center `abs(x - 0.32) < 0.12`, `abs(y + 0.40) < 0.10`, `0.07 < z < 0.22`, and linear speed below `0.12 m/s`. Its scripted grasp/lift/release/settle path has a headless MuJoCo execution check. This validates that demonstration path, not a learned policy or generalization.

## Recording lineage

[`control-ui/src/protocol/recorder.ts`](control-ui/src/protocol/recorder.ts) exposes:

```ts
const recorder = createEpisodeRecorder();
recorder.start();
recorder.recordSnapshot(snapshot);
recorder.recordCommand(command, 'operator'); // or demo_controller / policy
recorder.mark('operator_note', 'Optional operator annotation');
const episode = recorder.exportEpisode(); // object, no download side effect
downloadEpisode(episode);                 // browser Blob JSON download
recorder.reset();
```

Schema `werescue.episode.v1` stores episode ID, start/export UTC times, duration, provenance, and typed events. Event sequence is authoritative for ordering. UTC supports cross-system correlation; monotonic elapsed milliseconds measure within-session intervals. `snapshot.simTime` stays separate. A command references its preceding observation and remains `executionStatus: 'requested'` until a future execution acknowledgment protocol is implemented.

Current provenance explicitly says browser simulation, scripted demo controller, no recorded model inference, no camera frames included, and no completed learning. It identifies status-derived segmentation and `snapshot.lastAction` as applied 8D MuJoCo actuator targets. Array snapshots are copied so later simulation updates do not overwrite recorded observations. A `policy` actor label alone is not evidence that a VLA produced the action.

`recordCommand` records requests only. It does not change the segment or `controlSource`. Automatic boundaries come from observed simulator status transitions: `manual` opens recovery with operator control; an accepted return to `running` or `recovering` closes recovery, records `policy_handoff`, and starts a task segment under the demo controller. A rejected `resume_policy` request leaves subsequent `manual` snapshots in recovery. First running and completed transitions open/close task segments, while assistance, stop, and error transitions identify interruptions without claiming completion. An observed return to `ready` labels reset where applicable; reset must not count as recovery.

Command `actor` is the issuer of a request, and `controlSource` is the controller confirmed by observed simulator status. An operator-issued Start can therefore coexist with demo-controller execution. `mark(type, detail)` is annotation-only and cannot override either segmentation or controller ownership. An accepted simulator handoff still requires task/resumable-state evaluation before serving as a learning outcome label.

## Two learning objectives

| Dataset segment | Intended use | Completion label |
| --- | --- | --- |
| Failed state `3′` to resumable state `2` | Train a separate recovery VLA | Defined resumable-state predicate, with operator handoff time |
| Normal state `2` through task states `3` and `4` | Post-train the existing task VLA | Simulator-native task completion predicate |

Ghost trajectories are operator reference information. An old joint pose may be unsafe or insufficient if object contact or the surrounding scene changed. Restoring a simulator snapshot can create a comparable evaluation start, but cannot stand in for an executed recovery episode.

## Inputs still needed for VLA training

- Synchronized camera frames, view identity, calibration, frame timestamps, task instruction, robot identity, and simulator/version metadata.
- Action tensor definitions with units, frame, gripper semantics, control rate, and execution acknowledgments or applied controls.
- Source checkpoint and model configuration, data provenance, segment labels, success predicates, consent and training rights.
- An immutable split manifest separating training episodes, held-out normal starts, and held-out failure states.

The current UI snapshot exports neither camera pixels nor full physics state (`qpos`, `qvel`, actuator/internal state and other replay prerequisites). Exact state restoration/replay and synchronized visual training inputs therefore need additional export work.

Existing SmolVLA training work expects camera observations, state, and **7D actions**; the browser scene emits **8D actuator controls**. An explicit model-to-simulator adapter must reconcile action representation, robot/state ordering, gripper scaling, frames, units, and timing. Do not truncate controls or assume a shared array length is sufficient. Pixel capture, full-state replay, and this learning-policy bridge are the next integration tasks.

The current UI log alone is not a complete trainable VLA dataset. Do not fabricate observations, claim a scripted controller is SmolVLA, or report training as completed until actual logs and checkpoints exist.

## Evaluation and return artifacts

Compare task VLA before/after post-training using the same held-out starts. Separately compare task-only continuation against recovery VLA followed by task VLA on the same held-out failure states. Record final task successes and total attempts for each condition, plus recovery time, handoff success, human interventions, and safety stops where measurable.

Return dataset/split manifests, exact checkpoint identifiers, run configurations, training logs, evaluation JSON, and representative videos. Report measured results only. Simulation improvements do not establish real-robot safety, cross-site generalization, or customer demand. UI integration should consume explicit results and model status from a future adapter rather than inventing progress or scores.
