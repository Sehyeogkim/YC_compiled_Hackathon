<div align="center">

# RG Service Center

### When robots fail, get them back to work.

A remote recovery desk for robot operators — with intervention data for better policies.

**[Quick start](#quick-start) · [Demo walkthrough](#demo-walkthrough) · [Architecture](#architecture) · [Technical docs](#technical-docs)**

![Current RG Service Center UI: recorded SmolVLA replay, local human recovery controls, and episode recording](docs/assets/control-desk.png)

</div>

## The problem

A failed robot task becomes an operations problem: someone has to inspect what happened, recover the robot, and get the task moving again. The intervention also contains useful knowledge that is easy to lose.

**RG Service Center brings remote recovery and intervention recording into one workflow.** Operators can inspect recent motion, take control, complete a recovery, and preserve the demonstration for future policy learning.

## Current presentation UI

The current presentation runs at **http://127.0.0.1:5173/?local=1**, with the header **RECORDED SMOLVLA → LOCAL HUMAN CONTROL** and the **SmolVLA → human control** panel shown above.

Replay the recorded base-policy motion, take control in the local MuJoCo workcell, mark the recovery/task boundary, and download human intervention records plus a video of the view. The recording identifies SmolVLA base weights with an explicit LIBERO input/output adapter; it is not a raw zero-shot model evaluation. Camera reconstruction and dataset qualification are still required before these local records can be used for VLA training.

## Native GPU implementation

- **Real VLA inference.** A pinned SmolVLA checkpoint runs on a remote GPU against native LIBERO simulation.
- **A browser recovery desk.** Inspect the workcell, view native cameras, and take over end-effector and gripper control.
- **Native intervention recording.** Capture two RGB cameras, robot state, executed actions, controller ownership, and phase boundaries.
- **Two learning paths.** Export recovery and task-continuation segments separately, with conversion tooling for LeRobot datasets.

The current task uses a **Franka Panda** to **pick up an alphabet soup can and place it in a basket**. The browser's factory background is visual context; native LIBERO owns task physics, policy observations, and success detection.

## Quick start

```sh
git clone https://github.com/Sehyeogkim/YC_compiled_Hackathon.git
cd YC_compiled_Hackathon/control-ui
npm ci
npm run dev
```

Requires **Node.js 20.19+ or 22.12+** and WebGL2. After preparing the native scene cache, open **http://127.0.0.1:5173/?local=1** for the current presentation UI. See [local presentation setup](docs/LOCAL_DEMO.md) for the scene prerequisite. The large native scene cache is not committed to GitHub.

For the separate live GPU desk, open the base URL without `?local=1` and follow the [native GPU setup guide](docs/SETUP.md). A scripted interface preview is also available there under **Connection settings → Offline demo**.

## Demo walkthrough

### Current local presentation

1. Select **Play recorded SmolVLA** to replay saved native policy states.
2. Select **Take Control · record human** to begin local human recovery.
3. Use mouse/keyboard controls to move the robot and operate its gripper.
4. Select **Recovery complete · keep control**, then finish the task manually.
5. Select **Finish & download** to save the intervention and captured view.

The local control clock targets 20 Hz. This mode uses recorded policy motion; it does not run live GPU inference or establish a learned recovery success rate.

### Native GPU app · separate runtime

1. **Observe:** connect the GPU and select **Start VLA** to run the pretrained policy.
2. **Take over:** when you observe a failure, select **Take control**. Queued policy actions are cleared.
3. **Recover:** use mouse or keyboard controls to move to a resumable state.
4. **Mark the boundary:** select **Recovery complete → task**, keeping the human in control.
5. **Finish and record:** complete the task, then end the episode. Native LIBERO supplies the task-success predicate.

One intervention can yield two distinct segments:

| Segment | Demonstration | Learning objective |
| --- | --- | --- |
| **Recovery · R′** | Failed state → resumable state | Train a recovery policy |
| **Task continuation · R** | Resumable state → completed task | Improve the task policy |

Both proposed fine-tuning branches start independently from the same pretrained policy **P**. Original episode lineage is retained across exports.

## Architecture

The diagram below describes the **native GPU path**. The current `?local=1` presentation replays saved policy states and runs human control in browser MuJoCo.

![Technical architecture: browser desk, native GPU runtime, intervention records, and planned learning paths](docs/assets/architecture.png)

The browser sends operator commands to the FastAPI bridge. SmolVLA and human controls share one native LIBERO simulator; its observations and executed actions feed the intervention records. Qualified recovery and task segments form separate datasets for the planned training branches. World Labs supplies the browser background only.

| Layer | Stack |
| --- | --- |
| Operator interface | React · TypeScript · Vite |
| 3D visualization | Three.js · MuJoCo WASM · Spark |
| Factory background | World Labs Marble 1.1 Gaussian splat |
| Robot API | Python · FastAPI |
| Policy inference | SmolVLA · Hugging Face LeRobot · PyTorch · CUDA |
| Native simulation | LIBERO · MuJoCo · Franka Panda |

## Prototype status

**Implemented:** pretrained inference, native simulation, human takeover, intervention recording, segment export, and dataset-conversion tooling.

**Recorded verification:** [20 actual policy-driven native steps on an NVIDIA A40](inference/proof/native-policy-smoke.json). This is integration evidence; it does not establish a task-success rate or learned recovery improvement.

**Next:** qualify human demonstrations, fine-tune separate task and recovery policies, and compare them on held-out native starts. Automatic failure diagnosis, learned recovery, technician dispatch, and real-robot deployment are future work. Failure recognition and the recovery boundary currently rely on the operator.

## Technical docs

| Guide | Contents |
| --- | --- |
| [Setup and prerequisites](docs/SETUP.md) | Local UI, pinned GPU runtime, SSH tunnel |
| [Current presentation UI](docs/LOCAL_DEMO.md) | Recorded native states, local recovery, episode download |
| [Native control desk](control-ui/README.md) | Interface modes, controls, recording semantics |
| [Inference service](inference/README.md) | Model pins, runtime configuration, smoke checks |
| [API protocol](inference/PROTOCOL.md) | Commands, state, cameras, control ownership |
| [Collection and export](inference/COLLECTION_HANDOFF.md) | Native records and segment qualification |
| [Learning and evaluation](training/README.md) | R / R′ experiments and held-out comparisons |
| [Third-party notices](control-ui/THIRD_PARTY_NOTICES.md) | Asset sources and upstream licenses |

## Repository map

```text
control-ui/          Current local presentation, native GPU desk, offline simulation
inference/           GPU API, native simulation, recording, segment export
training/            Dataset conversion, fine-tuning, evaluation utilities
docs/                Setup guide and presentation assets
```

Internal runtime identifiers retain the earlier `werescue` name for compatibility. The product is **RG Service Center**.
