#!/usr/bin/env bash
set -euo pipefail
eval_root="${1:-/workspace/werescue-evaluation}"
runtime_python="${2:-/workspace/werescue-ui-inference/runtime/.venv/bin/python}"
pid_file="$eval_root/hfvla50.pid"
if [[ -e "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  echo 'HFVLA evaluation already running; inspect progress.json' >&2
  exit 2
fi
setsid env HF_HOME=/opt/hf MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  "$runtime_python" "$eval_root/training/evaluation/run_hfvla50.py" \
  --policy "$eval_root/models/HFVLA" \
  --assets /workspace/werescue-ui-inference/assets \
  --suite object --task-ids 0 --initial-states 0-24 --seeds 1000,1001 \
  --camera-size 360 --position-jitter 0.04 \
  --out "$eval_root/runs/hfvla-can-dev50" \
  > "$eval_root/hfvla50.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$pid_file"
printf 'Started HFVLA can development evaluation PID %s; 50 planned trials.\n' "$(cat "$pid_file")"
