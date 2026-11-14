#!/usr/bin/env bash
# One-shot setup for a Linux machine or VM with an NVIDIA GPU (also works CPU-only).
#
#   ./setup.sh                 # core + LeRobot (sims, datasets, SmolVLA/ACT/pi0...) + LoRA
#   ./setup.sh --minimal       # core only (reach2d sim, native policies, SmolVLM backbones)
#   ./setup.sh --smoke         # ...then run the reach_smoke pipeline end to end
#
# What it does: apt packages for headless rendering + video (if sudo is available), installs uv,
# creates .venv with `uv sync`, makes sure the torch build matches your NVIDIA driver, runs `vlab doctor`.
set -euo pipefail
cd "$(dirname "$0")"

EXTRAS=(--extra lerobot --extra peft --extra dev)
SMOKE=0
for a in "$@"; do
  case "$a" in
    --minimal) EXTRAS=(--extra dev) ;;
    --smoke) SMOKE=1 ;;
    --edge) EXTRAS+=(--extra edge) ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown flag $a"; exit 1 ;;
  esac
done

say() { printf '\033[1m==> %s\033[0m\n' "$*"; }

[[ "$(uname -s)" == "Linux" ]] || { echo "this script targets Linux"; exit 1; }

# ---------------------------------------------------------------- GPU
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  say "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) (driver $DRIVER)"
else
  DRIVER=""
  say "No working NVIDIA driver found - continuing in CPU mode."
  echo "    On Ubuntu: sudo ubuntu-drivers install && sudo reboot   (or use a GPU image with drivers)"
fi

# ------------------------------------------------------- system packages
if command -v apt-get >/dev/null 2>&1; then
  PKGS=(libegl1 libgl1 libopengl0 libglib2.0-0 ffmpeg git build-essential python3-dev curl)
  MISSING=()
  for p in "${PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p"); done
  if ((${#MISSING[@]})); then
    if sudo -n true 2>/dev/null; then
      say "apt install ${MISSING[*]}"
      sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${MISSING[@]}" >/dev/null
    else
      echo "    missing packages (install with sudo): ${MISSING[*]}"
    fi
  fi
fi

# EGL on /dev/dri needs the render group (takes effect on next login; rendering still works without it)
if [[ -e /dev/dri/renderD128 ]] && ! id -nG | grep -qw render && sudo -n true 2>/dev/null; then
  sudo usermod -aG render "$USER" || true
fi

# Compute-only driver images (e.g. GCP "ubuntu-accelerator") ship CUDA but not NVIDIA's EGL/OpenGL
# libraries, so MuJoCo silently renders on the CPU (ALOHA sim: ~170 ms/step instead of ~10 ms).
# Install the GL package that matches the installed driver exactly.
if [[ -n "$DRIVER" && ! -e /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]] && command -v dpkg >/dev/null 2>&1; then
  COMPUTE_PKG=$(dpkg-query -W -f='${Package}\n' 'libnvidia-compute-*' 2>/dev/null | head -1)
  GL_PKG=${COMPUTE_PKG/compute/gl}; GL_PKG=${GL_PKG%%:*}
  if [[ -n "$GL_PKG" ]] && sudo -n true 2>/dev/null; then
    say "installing $GL_PKG (GPU EGL rendering for MuJoCo)"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "$GL_PKG" >/dev/null || echo "    could not install $GL_PKG"
  else
    echo "    NVIDIA EGL missing: install the libnvidia-gl package matching driver $DRIVER for GPU rendering"
  fi
fi

# -------------------------------------------------------------------- uv
if ! command -v uv >/dev/null 2>&1; then
  say "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  export PATH="$HOME/.local/bin:$PATH"
fi
say "uv $(uv --version | cut -d' ' -f2)"

# ------------------------------------------------------------------ venv
say "uv sync ${EXTRAS[*]}"
uv sync "${EXTRAS[@]}"

# torch wheels on PyPI target a recent CUDA; older drivers need a matching build.
if [[ -n "$DRIVER" ]]; then
  if ! uv run --no-sync python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    say "torch cannot see the GPU with driver $DRIVER - installing a matching build"
    uv pip install --torch-backend=auto --reinstall torch torchvision
    echo "    note: use 'uv run --no-sync ...' or 'source .venv/bin/activate' so uv keeps this build"
  fi
fi

# headless MuJoCo / PyOpenGL rendering
grep -qs MUJOCO_GL .venv/bin/activate || cat >> .venv/bin/activate <<'EOF'
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
EOF

say "vlab doctor"
uv run --no-sync vlab doctor

if ((SMOKE)); then
  say "running reach_smoke (all eight stages)"
  uv run --no-sync vlab run reach_smoke
fi

cat <<'EOF'

Done. Next:
  source .venv/bin/activate
  vlab run reach_smoke              # every stage, tiny backbone (minutes)
  vlab run reach_full               # SmolVLM2 backbone (GPU, ~30 min)
  vlab run pusht_lerobot            # LeRobot data + sim + published baseline
  vlab list                         # everything you can swap
EOF
