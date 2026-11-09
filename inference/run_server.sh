#!/usr/bin/env bash
set -euo pipefail
app_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_root="${WERESCUE_RUNTIME:-$app_root/runtime}"
runtime_python="$runtime_root/.venv/bin/python"
[[ -x "$runtime_python" ]] || { echo 'Pinned inference runtime is not installed' >&2; exit 2; }
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export WERESCUE_DEVICE="${WERESCUE_DEVICE:-cuda}"
export WERESCUE_TORCH_THREADS="${WERESCUE_TORCH_THREADS:-4}"
export WERESCUE_HORIZON="${WERESCUE_HORIZON:-1000}"
export WERESCUE_POLICY_PATH="${WERESCUE_POLICY_PATH:-$app_root/models/P}"
export WERESCUE_RECORD_ROOT="${WERESCUE_RECORD_ROOT:-$app_root/runs}"
export WERESCUE_LIBERO_ASSETS="${WERESCUE_LIBERO_ASSETS:-$app_root/assets}"
export HF_HOME="${HF_HOME:-/opt/hf}"
cd "$app_root"
if [[ -e server.pid ]] && kill -0 "$(cat server.pid)" 2>/dev/null; then
  echo 'Inference server is already running; inspect server.log instead of launching another.' >&2
  exit 2
fi
setsid "$runtime_python" "$app_root/server.py" > "$app_root/server.log" 2>&1 < /dev/null &
server_pid=$!
printf '%s\n' "$server_pid" > "$app_root/server.pid"
printf 'Started inference server PID %s\n' "$server_pid"
