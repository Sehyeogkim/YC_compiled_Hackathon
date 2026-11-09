#!/usr/bin/env bash
set -euo pipefail
app_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_python="$app_root/runtime/.venv/bin/python"
[[ -x "$runtime_python" ]] || { echo 'Inference runtime missing' >&2; exit 2; }
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export WERESCUE_DEVICE=cuda
export WERESCUE_TORCH_THREADS=4
export WERESCUE_MODEL_ARM=base-adapted
export WERESCUE_HOST=127.0.0.1
export WERESCUE_PORT=8011
export WERESCUE_HORIZON=1000
export WERESCUE_POLICY_PATH="$app_root/models/P"
export WERESCUE_BASE_POLICY_PATH=/opt/hf/hub/models--lerobot--smolvla_base/snapshots/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6
export WERESCUE_POSITION_JITTER_M=0.04
export WERESCUE_RECORD_ROOT="$app_root/runs-base-adapted"
export WERESCUE_LIBERO_ASSETS="$app_root/assets"
export HF_HOME=/opt/hf
cd "$app_root"
if [[ -f base-server.pid ]] && kill -0 "$(cat base-server.pid)" 2>/dev/null; then
  echo 'Base collection server already running' >&2
  exit 2
fi
setsid "$runtime_python" "$app_root/server.py" > "$app_root/base-server.log" 2>&1 < /dev/null &
base_server_pid=$!
printf '%s\n' "$base_server_pid" > "$app_root/base-server.pid"
printf 'Started base collection server PID %s on port 8011\n' "$base_server_pid"
