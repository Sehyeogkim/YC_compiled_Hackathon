#!/usr/bin/env bash
# Isolated inference runtime; does not alter image /opt/lr or the training Pod.
set -euo pipefail
runtime_root="${1:-/workspace/werescue-ui-inference/runtime}"
framework_sha=b9cb121cb7d4e3e26ec5c906d08dda68d151ba52
mkdir -p "$runtime_root"
if [[ ! -e "$runtime_root/.venv" ]]; then
  /usr/local/bin/python -m venv --system-site-packages "$runtime_root/.venv"
fi
runtime_python="$runtime_root/.venv/bin/python"
framework_root="$runtime_root/lerobot"
if [[ ! -e "$framework_root" ]]; then
  git init "$framework_root"
  git -C "$framework_root" remote add origin https://github.com/huggingface/lerobot.git
  git -C "$framework_root" fetch --depth 1 origin "$framework_sha"
  git -C "$framework_root" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$framework_root" rev-parse HEAD)" == "$framework_sha" ]] || { echo 'Framework revision mismatch' >&2; exit 2; }
cat > "$runtime_root/constraints.txt" <<'EOF'
torch==2.8.0
torchvision==0.23.0
torchcodec==0.7.0
hf-libero==0.1.4
mujoco==3.3.7
EOF
"$runtime_python" -m pip install --disable-pip-version-check -c "$runtime_root/constraints.txt" \
  -e "$framework_root[smolvla,libero]" 'fastapi>=0.115,<1' 'uvicorn>=0.30,<1' 'pillow>=10,<12'
"$runtime_python" -m pip freeze > "$runtime_root/resolved-requirements.txt"
