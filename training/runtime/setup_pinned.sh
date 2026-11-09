#!/usr/bin/env bash
# Optional clean setup. First try native_smoke.py in the existing RunPod image.
# Does not delete or alter an existing framework checkout or environment.
set -euo pipefail
runtime_root="${1:-/workspace/werescue-runtime}"
framework_sha=b9cb121cb7d4e3e26ec5c906d08dda68d151ba52
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Native LIBERO setup requires Linux. No changes made.' >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  echo 'Install uv on the RunPod container before running this optional setup.' >&2
  exit 2
fi
mkdir -p "$runtime_root/src"
framework_root="$runtime_root/src/lerobot"
if [[ ! -e "$framework_root" ]]; then
  git clone --no-checkout https://github.com/huggingface/lerobot.git "$framework_root"
  git -C "$framework_root" checkout --detach "$framework_sha"
else
  actual_sha="$(git -C "$framework_root" rev-parse HEAD)"
  if [[ "$actual_sha" != "$framework_sha" ]] || [[ -n "$(git -C "$framework_root" status --porcelain)" ]]; then
    echo 'Existing checkout differs or has edits; choose a different runtime directory.' >&2
    exit 2
  fi
fi
uv python install 3.12
if [[ ! -e "$runtime_root/.venv" ]]; then
  uv venv --python 3.12 --seed "$runtime_root/.venv"
fi
runtime_python="$runtime_root/.venv/bin/python"
"$runtime_python" -c 'import sys; assert sys.version_info[:2] == (3,12), sys.version'
# Match torchcodec to torch; all remaining resolved versions are frozen below.
cat > "$runtime_root/constraints.txt" <<'EOF'
torch==2.8.0
torchvision==0.23.0
torchcodec==0.7.0
hf-libero==0.1.4
mujoco==3.3.7
EOF
uv pip install --python "$runtime_python" -c "$runtime_root/constraints.txt" \
  -e "$framework_root[smolvla,libero,training,evaluation]"
uv pip freeze --python "$runtime_python" > "$runtime_root/resolved-requirements.txt"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
"$runtime_python" -c 'import ctypes; ctypes.CDLL("libEGL.so.1"); ctypes.CDLL("libGL.so.1")'
"$runtime_python" "$script_root/download_checkpoint.py" --out "$runtime_root/models/P"
{
  printf 'export WERESCUE_RUNTIME=%q\n' "$runtime_root"
  printf 'export PATH=%q:"$PATH"\n' "$runtime_root/.venv/bin"
  printf 'export MUJOCO_GL=egl\nexport PYOPENGL_PLATFORM=egl\nexport TOKENIZERS_PARALLELISM=false\n'
} > "$runtime_root/env.sh"
echo "Prepared optional isolated runtime. Activate with: source $runtime_root/env.sh"
