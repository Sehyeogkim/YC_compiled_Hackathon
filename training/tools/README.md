# Experiment bookkeeping tools

Python 3.10+; standard library only. No model, dataset, or measured result is bundled.

```sh
python3 -m unittest discover -s training/tests -v
python3 training/tools/manifests.py collection /path/to/collection.json --output training/artifacts/segments.json
python3 training/tools/evaluate.py /path/to/measured_trials.json --output training/artifacts/task_summary.json --svg training/artifacts/task_success.svg
python3 training/tools/manifests.py recovery-video /path/to/recovery_video.json --output training/artifacts/recovery_video_validated.json
```

## Collection input

Top-level `schema_version` is `werescue.collection.v1`; `episodes` is a nonempty list.
Each episode requires:

- Identity: `episode_id`, `parent_episode_id` (original intervention lineage), `initial_state_id`, `split` (`train`, `val`, `test`), `task_id`, `environment_id`, `robot_id`, `operator_id`.
- Recording: `frame_count`, positive `fps`, `action_schema` describing native actions, `actions_are: "executed"`.
- Local files: `camera_refs` (map of camera name to video/frame archive path), `state_ref`, `action_ref`, `timestamps_ref`. Paths resolve relative to input JSON. Actual files are required and SHA-256 hashed.
- `recovery` and `task` objects, each with `start_frame`, exclusive `end_frame`, `control_source: "operator"`, `success: true`, `success_predicate`, `predicate_verified: true`, `evidence_ref` (local independent predicate evidence), `contains_reset: false`.

Recovery ends exactly at the task start frame. The indexes select `R_prime = recovery` and `R = task` without rewriting actions or generating pixels. Shared parent episodes or initial-state IDs cannot cross splits. Failed/unsafe/unverified segments should be preserved in raw storage; this CLI rejects them as successful imitation targets.

Validation covers metadata, phase boundaries, split lineage, and asset existence/hash. It does **not** decode pixels, verify aligned tensor lengths/units, certify that supplied labels are true, or establish native simulator replay. A native adapter must perform those checks before training. UI snapshots without camera pixels and executed native actions cannot pass by themselves.

## Paired task evaluation input

Top-level `schema_version: "werescue.task_eval.v1"`, nonempty `protocol_id`, and `trials` list. Each trial requires:

- `arm`: `P` or `P+R`.
- `environment_id`, `task_id`, `start_state_id`, integer `seed` (paired key).
- `checkpoint` (fixed per arm and distinct between arms), `trace_ref` (trace identifier/path).
- `split: "test"`, `control_source: "policy"`, `contains_human_intervention: false`.
- `success` boolean and `success_predicate_verified: true` from the environment-native predicate.

Optional `training_initial_state_ids` enables an additional training-state exclusion check; omission is disclosed in the summary. Both arms must contain exactly the same paired starts/seeds, including failures/timeouts, without duplicates. The evaluation runner owns native execution and truthful trace/outcome provenance; the summary tool validates supplied records, not policy execution.

The output includes successes/attempts, Wilson 95% intervals, change in percentage points, and paired outcomes (`00`, `01`, `10`, `11`: baseline/finetuned failure=0, success=1). SVG uses a fixed 0–100% scale. It only renders nonempty measured trial input; no example performance graph is generated. Per-arm intervals are not a statistical test of the paired difference.

## Recovery video input

Top-level `schema_version: "werescue.recovery_video.v1"`; fields: `checkpoint`, `failure_state_id`, `recovery_state_id`, `task_id`, `recovery_predicate`, `control_source: "policy"`, `contains_reset: false`, `recovery_predicate_passed: true`, `split` (`demo` or `test`), numeric `start_seconds`/`end_seconds`, and local `video_ref`, `execution_trace_ref`, `predicate_evidence_ref`.

The video interval should show only failed state `3′` to verified resumable state `2`. The validator hashes referenced files and rejects reset/scripted/operator recovery records. Metadata validation does not watch the video or prove its labels. One clip demonstrates one recorded recovery and does not establish a recovery success rate; use `demo` unless held-out lineage is verified externally.
