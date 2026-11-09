# WeRescue learning setup

This directory belongs to the learning thread. Do not edit the concurrent `control-ui/` implementation from here.

## Agreed experiment

| Symbol | Meaning | Output |
|---|---|---|
| P | pinned `lerobot/smolvla_libero` checkpoint | frozen baseline |
| R | human task continuation, `2 → 3 → 4` | fine-tune P independently; compare autonomous task completion against P |
| R′ | human recovery, `3′ → 2` | fine-tune P independently; record actual policy recovery video |

One fully qualified human intervention can produce both segments. Target 10 human interventions to collect, with the human performing both recovery and task completion. Qualification is checked afterward: the resulting usable dataset may contain fewer than 10 R′ or R segments. One source episode yields at most one of each segment, never two independent interventions. Returning to automatic control at state 2 would produce a policy suffix, not a human R demonstration. Keep the raw full intervention and an explicit phase boundary.

`P+R` means fine-tuning P on R. It does not mean replaying the original pretraining data. `P+R′` starts from the same original P, not from `P+R`.

## Environment decision — 2026-11-09

Use **the actual native LIBERO environment** for rendering, collection, training and evaluation. Replace the factory scene in the demo with that environment for this experiment. Preserve the control-desk product interaction around it. Do not reskin one engine while training on a different one and claim the results transfer.

Initial candidate task: `libero_object`, task 0, `pick_up_the_alphabet_soup_and_place_it_in_the_basket`. Runtime must verify the task name before using an index. Select the final task/condition using a small development pilot **before collecting the 10 attempted interventions**. Prefer P success around 40–70%, with one recurring human-correctable failure type. This band is an engineering selection target, not a measured result. Avoid both ceiling performance and almost no baseline ability. Hold standard policy/controller settings constant; do not degrade P solely to produce a comparison. The final test pool stays excluded from task and checkpoint selection. This is a single-task fine-tuning experiment, not the full LIBERO benchmark.

Model lock and first-pass hyperparameters are in [experiment.json](experiment.json). Task and recovery training each start independently from `lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d`. The checkpoint expects state 8, action 7, and policy camera keys `camera1/camera2`; the native human dataset keys `image/image2` are renamed exactly at training input. The `n_action_steps=10` override, `num_steps=10`, chunk size 50, and 256×256 native rendering match the current Control UI P and evaluation. The Control UI records upright PNGs for display; the human training converter restores the raw policy-view orientation by flipping H and W.

The Control UI now uses this same pinned P and NAS10 setting. The converter records the policy that produced each failure per segment, and the training launcher discloses any mismatch. A separate HFVLA 20-step self-imitation practice run remains a pipeline test and is excluded from the P, P+R, and P+R′ comparison.

## Collection and success boundaries

1. Obtain a reproducible failed native state. Preserve its originating episode/start ID, simulator version, full state, task instruction and failure source. An injected perturbation must be labeled as injected.
2. Before recording, restore that state as the episode's start. This restoration is not a recovery action.
3. Record human recovery using synchronized RGB, pre-action state, applied native action, next state, timestamps and full-state checkpoints.
4. At the recovery endpoint, independently check open gripper, safe resumable TCP/object/contact conditions, and stability; historical joint-pose similarity alone is insufficient.
5. Mark the transition while retaining human control. Record human task completion, using the simulator's native task success predicate.
6. Slice R′ and R into separate episodes so action chunks never cross the handoff boundary. Retain original episode lineage and phase tags.

The Control UI now records native state8, executed relative OSC action7, and both 256×256 camera PNGs. `inference/export_segments.py` plus `training/runtime/convert_human_segments.py` can prepare separate LeRobot R′ and R episodes after qualification. The recovery endpoint remains an operator annotation unless independent predicate evidence is supplied; report that scope. Never truncate 8D joint actuator targets into 7D policy actions.

## Evaluation

P and P+R run exactly the same 50 held-out trials (25 native layouts × 2 policy RNG seeds), with the same seed, camera/controller settings, success predicate and time budget. The current allocation reserves native initial-state IDs 0–24 for collection/development and 25–49 for final tests. Target 10 collected interventions across the collection layouts, allowing independent repeated demonstrations; report the actual qualified count for each arm. Final comparison uses 25 layouts × policy RNG seeds 1000 and 1001 = 50 attempts per arm. Recovery-video starts 45–49 are a subset of that test pool, not extra independent test examples. Show actual `successes / attempts` and confidence intervals, including timeouts. These starts are held out from **intervention fine-tuning**, not necessarily from the original P pretraining data. A fixed final-step checkpoint prevents selecting a checkpoint using test success; collect a separate validation pool before tuning checkpoints.

P+R′ video starts at a failure snapshot excluded from its training lineage. Run the learned recovery policy, stop when the resumable-state predicate passes, and retain action trace, checkpoint ID and predicate evidence. One successful clip demonstrates that execution; it is not an estimated recovery success rate.

Measured-result and segment validation tools are documented in [tools/README.md](tools/README.md). They do not fabricate graphs, run training, or certify native replay. One separate P self-imitation **practice** checkpoint exists on RunPod; it is not human R/R′ training. No human training episodes, P+R or P+R′ checkpoints, or measured intervention-improvement rates exist yet.

## RunPod

The user chose RunPod for the actual GPU work. Native rendering and policy smoke must pass before collecting the intervention dataset and starting the two training branches. Keep datasets, checkpoints, logs and videos on `/workspace` durable storage. GPU provisioning details and verified state belong in `runtime_status.json`.

Sources: [pinned model configuration](https://huggingface.co/lerobot/smolvla_libero/blob/a24f1f380461b665158d52acd140152c79e8b36d/config.json), [LeRobot LIBERO integration](https://huggingface.co/docs/lerobot/main/en/libero).
