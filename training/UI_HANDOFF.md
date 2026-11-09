# Native LIBERO scene replaces factory simulation for the learning demo

User decision in the learning thread, 2026-11-09: due to time limits, replace the custom factory/conveyor rendering with the **existing native LIBERO task environment** and collect, train and evaluate there. Keep the service-center Control UI concept. GPU host: RunPod.

- Selected first task: `libero_object`, task 0, alphabet soup can → basket; verify task name at runtime.
- Observation cameras must come from the same authoritative native simulator that steps the robot, not independent browser geometry.
- The task policy is the pinned `lerobot/smolvla_libero` checkpoint; 8D observation state and 7D relative OSC actions, two RGB cameras.
- Human intervention yields two segments: R′ = failed `3′` → safe resumable `2`; R = human continuation `2` → completed `4`.
- **The phase marker at 2 must keep the human in control** when collecting R. Returning to P at that point makes the suffix policy-generated, not human R data.
- Do not use browser joint actuator controls as LIBERO policy actions. A native bridge is required.
- Product workflow: monitor native camera → failure → human takeover → recovery marker → human task completion → two dataset segments. Learned recovery later uses separate P+R′ inference.
- Training thread writes only `training/`; UI owner can use this decision without concurrent edits from this thread.

See `experiment.json`, `README.md`, and `runtime/` for exact training-side contracts and scripts. A ready Pod alone is not a verified working native simulator/policy/collector.
