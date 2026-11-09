# Standalone native LIBERO baseline evaluation

This process owns an independent native simulator. It does **not** call the interactive HTTP service or reset the operator's session. Two standalone processes can share a GPU if memory permits; they compete for throughput, so these runs are success-rate tests rather than inference-speed benchmarks.

## Arms and labels

- `libero`: exact `lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d` weights and processors.
- `base-adapted`: exact `lerobot/smolvla_base@5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6` weights, with the explicitly declared native LIBERO state/action/camera contract and LIBERO checkpoint normalization processors. No gradient updates. **This is an adapted-base diagnostic, not raw zero-shot performance.** The helper records this distinction in runtime provenance.
- Raw unadapted `smolvla_base` exposes state6/action6/three cameras and SO100 normalization data. Its native LIBERO success rate is **N/A**, not a fabricated 0%.

## Pilot: task selection first

```sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false

python training/evaluation/run_baseline.py \
  --arm libero --policy /workspace/werescue-ui-inference/models/P \
  --assets /workspace/werescue-ui-inference/assets \
  --out /workspace/werescue-evaluation/runs/libero-pilot --phase pilot

python training/evaluation/run_baseline.py \
  --arm base-adapted --policy /opt/hf/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 \
  --calibration-policy /workspace/werescue-ui-inference/models/P \
  --assets /workspace/werescue-ui-inference/assets \
  --out /workspace/werescue-evaluation/runs/base-adapted-pilot --phase pilot
```

Pilot defaults: native initial-state IDs0–9 × policy RNG seeds1000/1001 = **20 trials per arm**. Choose tasks/conditions using pilot only. The `--suite` and `--task-id` options default to LIBERO-Object task0, alphabet soup can into basket.

## Final: 50 runs per arm

Use distinct output directories and `--phase final`. Defaults are IDs25–49 × policy RNG seeds1000/1001 = **50 trials per arm across 25 native layouts**. Two RNG repetitions do not represent 50 independently sampled layouts. These starts are held out from intervention tuning, not claimed unseen during original checkpoint pretraining. Do not change the task or training configuration based on final-test results.

The frozen protocol is native initial states enabled, hard resets between trials, relative OSC at20Hz, horizon280 actions, chunk50, execute10 actions before replanning, native `check_success()`, no human interventions and no resets during a rollout. Environment seed is `10000+initial_state_id` and is identical across policy RNG seeds/arms. Both arms clip postprocessed normalized actions to[-1,1] and retain both the unclipped request and applied simulator action. Model weights load once per process; one native environment is active per process.

`--initial-states` accepts a reserved subset for diagnostic runs, e.g. `--initial-states 0-1 --phase pilot`. Final selection cannot include training/pilot IDs. `--seeds` must remain1000,1001 and `--horizon` must remain280 in this frozen protocol.

## Evidence and resume

- `protocol.json`: full selection and configuration, hashed. Different protocols cannot share an output directory.
- `runtime.json`: framework package/source hashes, checkpoint config digest, asset pin, GPU and adaptation metadata.
- Per trial: exact initial native state NPZ and numerical content hash, XML+hash, initial object positions, terminal native state, gzip action trace and atomic `done.json` measured outcome.
- First observed success and first observed failure retain representative side-by-side native-camera videos. Other trial videos are discarded after completion. There is one more terminal observation frame than actions.
- `summary.json`: continuously updated measured successes/attempts and descriptive Wilson interval. Repeated seeds share layouts; the interval does not account for layout clustering.

Only a matching valid `done.json` can be skipped. Resume rehashes trace/state/XML files and validates protocol/checkpoint/trial metadata. An incomplete trial is rerun from its initial state. Changed source/runtime versions cannot be combined with earlier completed trials in the same directory. Per-layout policy seed repetitions require the identical numerical native starting-state hash. Compare those hashes across arms as well before reporting a matched comparison.

This evaluator records actual policy execution. It never counts an incompatible model as a failed rollout, invents rates, fills missing trials, substitutes teleoperation, or silently wraps unavailable initial-state IDs. It imports only `load_policy_stack`/recording helpers from `runtime/native_smoke.py`; it deliberately uses the verified new `preprocess_observation→task_description→env_preprocessor` path rather than that older CLI's removed helper.

## Local contract checks

```sh
python -m unittest discover -s training/evaluation -p 'test_*.py' -v
python training/evaluation/run_baseline.py --help
```

These tests check selection, split leakage, resume evidence validation and adapter metadata. They do not establish GPU/runtime execution or task success.
