# RG Service Center · Control UI

Hackathon remote robot recovery desk with native GPU inference and a separate offline browser simulation. The initial customer is a robot supplier remotely managing deployed robots across customer factories and logistics centers. The demonstration focuses on incident observation, human recovery, task resumption, and an inspectable episode record.

## Run locally

Use Node.js 20.19+ or 22.12+ and npm. From the repository root:

```sh
cd control-ui
npm install
npm run dev
```

Open the local URL printed by Vite, normally `http://localhost:5173`. Build and serve the production bundle with:

```sh
npm run build
npm run preview
```

The build runs TypeScript before Vite. A successful build checks packaging and types; it does not verify robot recovery performance.

## What the demo demonstrates

### Native GPU mode

The **GPU · actual VLA** mode connects to the [native inference service](../inference/README.md), which runs the pinned `lerobot/smolvla_libero` checkpoint in native LIBERO. The GPU simulator owns physics, task outcomes, camera observations, state and executed 7D relative OSC actions; browser 3D visualizes that state. Select this mode, connect the GPU, and start VLA inference after following the [root setup guide](../README.md).

On a failure, the operator takes control, uses Ghost history as a reference, recovers, selects **Recovery complete → task**, and finishes the task manually. This boundary is an operator annotation, and keeps human control so the two segments remain human recovery R′ and human task continuation R. Native LIBERO's success predicate determines task completion. The service records synchronized native camera PNGs, pre-action state and executed actions on the GPU. See the [collection guide](../inference/COLLECTION_HANDOFF.md) and [API protocol](../inference/PROTOCOL.md).

A trained recovery VLA, automatic VLM failure analysis, and human-intervention improvement results are not demonstrated by this UI. Recovery currently comes from the operator.

### Offline demo

The **Offline demo** mode uses a scripted IK controller in browser MuJoCo. It runs without the inference GPU. Its movements and exported interaction logs are separate from native policy rollouts and native human training data.

The offline interaction is: start a task, inject a failure, pause for an operator, take manual control, move to a resumable state, and resume the demo controller. The episode log records commands and observations. A simulator reset restores state and is explicitly labeled as a reset; it is not an executed robot recovery.

The browser loads [`public/models/panda/factory.xml`](public/models/panda/factory.xml) and local assets directly. The conveyor uses a velocity/orientation servo approximation to feed the free can before stopping for contact-based grasping; it is not a calibrated belt-contact model. Asset provenance and licenses are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Offline command and snapshot contract

The canonical TypeScript contract lives in [`src/protocol/types.ts`](src/protocol/types.ts). The UI sends `SceneCommand` values to the scene, and the scene reports `SimSnapshot` values back. A command `id` identifies a request. The scene should consume each ID once even when React rerenders.

Commands are `start`, `pause`, `reset`, `inject_failure`, `take_control`, `resume_policy`, `stop`, `move`, and `gripper`. A `move` carries a three-component Cartesian `delta` in meters in the MuJoCo Z-up world frame; the robot base is at `[0, 0, 0]`. A `gripper` command carries a normalized target (`0` closed, `1` open). The scene owns workspace constraints, command acceptance, and the applied actuator controls.

Snapshots include state-machine status, simulation time in **seconds**, joint positions in **radians**, end-effector position, gripper state, can and target positions, retry count, task progress, belt state, a message, physics readiness, applied controls, and success. Positions use **meters in the Z-up world frame**. The belt runs along Y, and the bin center is approximately `[0.32, -0.40, 0.055]`. `eef` is the TCP site, with a 0.1034 m hand offset. Optional `stage` and `contactCount` add context. `physicsReady` reports the scene's engine state and does not establish physical accuracy.

`lastAction` is the current **8-component MuJoCo `data.ctrl` vector**: seven arm joint targets followed by a gripper control in `[0, 255]`. It is distinct from the operator's three-component jog delta and from measured joint motion. The snapshot's normalized `gripper` observation is finger displacement divided by 0.04 m. An error snapshot can have an empty `lastAction`.

The current scene evaluates success after its placement sequence: can-center `abs(x - 0.32) < 0.12`, `abs(y + 0.40) < 0.10`, `0.07 < z < 0.22`, and can linear speed below `0.12 m/s`. These are the implemented simulation predicates, not evidence of real-world containment or safety. The scripted grasp, lift, bin release and settling path has a headless MuJoCo execution check; this does not establish a learned recovery policy.

## Offline episode recording and export

```ts
import { createEpisodeRecorder, downloadEpisode } from './src/protocol/recorder';

const recorder = createEpisodeRecorder();
recorder.start();
recorder.recordSnapshot(snapshot);
recorder.recordCommand(command, 'operator');
recorder.mark('operator_note', 'Checked the can position before resuming.');
downloadEpisode(recorder.exportEpisode());
```

`start()` is idempotent and does not erase an existing episode. Call `reset()` explicitly before recording a new one. `exportEpisode()` returns a detached object. `downloadEpisode()` writes that object to a browser JSON download using a Blob.

Each event has an authoritative sequence number, UTC wall-clock timestamp, and monotonic milliseconds since recording began. `snapshot.simTime` preserves the simulation clock separately. Commands record their actor (`operator`, `demo_controller`, or `policy`), the current control source, and the preceding observation's sequence number. A requested command is not an acknowledged or executed action. Snapshot arrays are copied at capture time to preserve history.

Commands record **requests only** and never change the current segment or control source. An observed transition to `manual` opens recovery under operator control. A later observed transition from recovery to `running` or `recovering` records `recovery_end`, `policy_handoff`, and `task_start` under the demo controller. A rejected resume request followed by another `manual` snapshot keeps the recovery segment and operator control intact. An observed first task run opens `task_start`; completion closes `task_end`. Assistance, stop, and error transitions mark interrupted segments with explanatory details, without claiming completion. An observed return to `ready` records a simulation reset where applicable.

The request `actor` identifies who issued the command. `controlSource` identifies the controller confirmed by the observed simulator mode; these can differ when an operator clicks Start or Return to demo controller. Public `mark(type, detail)` adds annotations without changing segment or control source. Automatically inferred handoffs mean the simulator accepted a mode transition, not that a learned policy or real-world safety was validated.

Export provenance identifies browser simulation, the scripted controller, command-request semantics, status-derived segmentation, and `snapshot.lastAction` as the source of applied 8D MuJoCo actuator targets. It also declares the absence of included camera frames and completed learning. Preserve this provenance when passing records to another process. Do not relabel scripted episodes as human-only demonstrations or model-generated policy rollouts.

## Model and training integration

The native adapter is implemented in [`src/native/`](src/native/) and [`../inference/`](../inference/). It records model/checkpoint identity and separates requested commands from executed actions. [`../inference/export_segments.py`](../inference/export_segments.py) and [`../training/runtime/convert_human_segments.py`](../training/runtime/convert_human_segments.py) export and convert eligible human R′ and R segments into separate LeRobot episodes, preserving lineage and operator-annotation status. Qualified human datasets, P+R/P+R′ checkpoints and measured intervention gains remain pending; conversion tools alone do not establish learning results.

[`../TRAINING_HANDOFF.md`](../TRAINING_HANDOFF.md) retains the earlier browser-adapter requirements. Any new model adapter must define action mapping and control rate, acknowledge accepted/executed actions, synchronize camera frames, identify its checkpoint, and keep training and held-out evaluation states separate.

Human recovery segments and normal task segments serve different learning objectives. The offline browser JSON is an inspectable interaction log: it does not include camera pixels, camera calibration, or the full physics state needed for exact state restoration. Existing SmolVLA work expects camera observations, state, and 7-component actions; this scene reports 8-component actuator controls. Those formats require an explicit adapter with verified units, semantics, and timing. Do not drop one component and assume compatibility. The separate native service already records camera pixels and native state/action data. Exact simulator replay equivalence and learned recovery performance remain unverified.

## Current collection UI

See [UI_FLOW.md](UI_FLOW.md) for the agreed no-VLM operator flow. In the actual GPU mode, takeover opens Ghost history; recovery completion marks the boundary while keeping the human in control through final task completion. Reset/next episode follows the completed or ended episode. The API contract is in [../inference/PROTOCOL.md](../inference/PROTOCOL.md).

## Conveyor factory background

See [FACTORY_BACKGROUND.md](FACTORY_BACKGROUND.md) for the Marble 1.1 can-recycling environment, reference panorama, full-resolution loading, meter-scale calibration and generation tools.
