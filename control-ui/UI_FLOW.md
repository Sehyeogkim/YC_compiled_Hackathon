# Agreed operator flow

The current MVP excludes VLM-based detection and explanation. A human watches the actual VLA rollout and chooses when to intervene.

1. Connect the GPU and load the native LIBERO session.
2. Start actual SmolVLA inference. The policy sees the original native camera observations.
3. If the operator observes struggling, take control. Stop sending policy steps and clear queued policy actions.
4. Display the Ghost UI: actual earlier robot configurations and the observed EEF path. The operator selects a historical reference and checks the surrounding object/contact state. The reference is guidance, not a verified safe trajectory.
5. The operator executes recovery from failed state 3′ to resumable state 2.
6. Mark recovery complete. This changes the segment label, **not control ownership**.
7. The same operator completes the task from 2 through 3 to 4. Both recovery and successful task continuation therefore contain human actions. While human control is active, zero-motion hold steps keep native physics and recording running so the gripper can open, the can can fall, and the simulator can detect success. Stop loop pauses this stepping; Resume human control or a new operator input resumes it.
8. Finish using the native task outcome, then reset for the next run. If an intervention cannot be completed, choose **End & new episode**: the reset request and unfinished-segment status are retained in the previous GPU recording. This explicitly abandons that episode and starts a new one; it never labels the abandoned trajectory as success.

Recovery segment R′ trains the separate recovery policy. Human task-continuation segment R post-trains the task policy. Policy-generated continuation cannot be mislabeled as human R data.

## State ownership

The GPU LIBERO simulator owns all robot/object states, cameras, action execution, and task success. The Three.js native view loads the same MJCF/assets and displays received qpos/qvel using `mj_forward` only. Factory surroundings are presentation. The browser does not advance an independent simulator in native mode.

The separate Offline simulation uses local MuJoCo physics and a scripted controller. It is a UI/interaction demonstration and is not a VLA rollout or the native training dataset.

## Mouse controls

The cyan target ring moves the robot through operator action requests. Drag for XY and scroll over it for height. The current implementation is new code informed by Physical Arena's interaction semantics. Physical Arena source was not copied. The Ghost robot is a separate visual reference and is shown on takeover.

## Reset boundary

A native reset restores the episode's robot/object initial state and policy queues. It starts a new recording and is not an executed recovery action. Failed or unfinished interventions remain distinguishable from successful behavior-cloning examples.
