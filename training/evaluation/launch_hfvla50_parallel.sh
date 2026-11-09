#!/usr/bin/env bash
# Four independent processes, disjoint cases, one frozen model/task condition.
set -euo pipefail
eval_root="${1:-/workspace/werescue-evaluation}"
runtime_python="${2:-/workspace/werescue-ui-inference/runtime/.venv/bin/python}"
batch_root="$eval_root/runs/hfvla-can-dev50-parallel"
mkdir -p "$batch_root"
ranges=('0-5' '6-11' '12-17' '18-24')
for index in 0 1 2 3; do
  pid_file="$batch_root/worker-$index.pid"
  if [[ -e "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "Worker $index already running; do not duplicate cases" >&2
    exit 2
  fi
done
for index in 0 1 2 3; do
  setsid env HF_HOME=/opt/hf MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$runtime_python" "$eval_root/training/evaluation/run_hfvla50.py" \
    --policy "$eval_root/models/HFVLA" \
    --assets /workspace/werescue-ui-inference/assets \
    --suite object --task-ids 0 --initial-states "${ranges[$index]}" --seeds 1000,1001 \
    --camera-size 360 --position-jitter 0.04 \
    --out "$batch_root/worker-$index" \
    > "$batch_root/worker-$index.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$batch_root/worker-$index.pid"
  printf 'Started worker %s PID %s for native state IDs %s.\n' "$index" "$(cat "$batch_root/worker-$index.pid")" "${ranges[$index]}"
done
echo '12+12+12+14=50 planned trials. Partial or failed trials are not filled in.'
