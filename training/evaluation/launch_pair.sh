#!/usr/bin/env bash
# Run two standalone development pilots; never reset the interactive UI service.
set -euo pipefail
eval_root="${1:-/workspace/werescue-evaluation}"
runtime_python="${2:-/workspace/werescue-ui-inference/runtime/.venv/bin/python}"
libero_policy="${3:-/workspace/werescue-ui-inference/models/P}"
base_policy="${4:-/opt/hf/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6}"
assets="${5:-/workspace/werescue-ui-inference/assets}"
condition="${6:-fixed}"
case "$condition" in
  fixed) runner="$eval_root/training/evaluation/run_baseline.py"; prefix='' ;;
  position) runner="$eval_root/training/evaluation/run_position_pilot.py"; prefix='position-' ;;
  *) echo 'condition must be fixed or position' >&2; exit 2 ;;
esac
[[ -x "$runtime_python" && -f "$runner" ]] || { echo 'Runtime or evaluator missing' >&2; exit 2; }
mkdir -p "$eval_root/runs"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export HF_HOME=/opt/hf
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
for arm in libero base-adapted; do
  pid_file="$eval_root/$prefix$arm.pid"
  if [[ -e "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$arm already running; inspect its log instead of duplicating evaluation" >&2
    exit 2
  fi
done
setsid "$runtime_python" "$runner" --arm libero --policy "$libero_policy" \
  --assets "$assets" --phase pilot --out "$eval_root/runs/${prefix}libero-pilot" \
  > "$eval_root/${prefix}libero.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$eval_root/${prefix}libero.pid"
setsid "$runtime_python" "$runner" --arm base-adapted --policy "$base_policy" \
  --calibration-policy "$libero_policy" --assets "$assets" --phase pilot \
  --out "$eval_root/runs/${prefix}base-adapted-pilot" \
  > "$eval_root/${prefix}base-adapted.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$eval_root/${prefix}base-adapted.pid"
echo 'Launched both native development pilots; completion is summary.json with 20 completed trials per arm.'
