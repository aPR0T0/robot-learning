#!/usr/bin/env bash
# SAM 3D Objects (Meta) in its own conda env, following its doc/setup.md, the reconstruction half of real2sim.
#   scripts/real2sim/setup_sam3d.sh [ROOT]      (default ROOT=~/real2sim)
# Weights are gated: request access to facebook/sam-3d-objects (and facebook/sam3) on Hugging Face and
# log in (`huggingface-cli login`) first; the checkpoint step is skipped until a token can read them.
set -euo pipefail
source "$(dirname "$0")/common.sh"
ROOT=${1:-$HOME/real2sim}
SAM3D_REV=${SAM3D_REV:-$(pinned sam-3d-objects)}   # default: the thirdparty/ submodule's commit
mkdir -p "$ROOT" && cd "$ROOT"
MF=${MAMBA_ROOT:-$HOME/miniforge3}
set +u   # conda's activate scripts read unset variables
source "$MF/etc/profile.d/conda.sh"
clone_at sam-3d-objects https://github.com/facebookresearch/sam-3d-objects "$SAM3D_REV"
cd sam-3d-objects
conda env list | grep -q '^sam3d-objects ' || "$MF/bin/mamba" env create -q -y -f environments/default.yml
conda activate sam3d-objects
# doc/setup.md installs with pip; pip's resolver spends 20+ minutes backtracking over the ~150 pins, so use uv
# with the same pins. The NGC index in setup.md no longer resolves; the PyTorch cu121 index is enough.
export PATH="$HOME/.local/bin:$PATH" CUDA_HOME="$CONDA_PREFIX" MAX_JOBS=${MAX_JOBS:-$(nproc)}
PY="$CONDA_PREFIX/bin/python"
# nvidia-pyindex (in requirements.txt) only points pip at the NGC index: drop it with an override that never matches
printf 'nvidia-pyindex ; python_version < "0"\n' > /tmp/sam3d-overrides.txt
UVP=(uv pip install --python "$PY" --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match
     --override /tmp/sam3d-overrides.txt)
"${UVP[@]}" -e '.[dev]'
"${UVP[@]}" hatchling hatch-requirements-txt editables   # the --no-build-isolation builds below need the build backend
# [p3d]: flash-attn from its prebuilt wheel (a source build takes about an hour), PyTorch3D built against torch
"${UVP[@]}" "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
"${UVP[@]}" --no-build-isolation -e '.[p3d]'
# [inference]: kaolin from NVIDIA's wheel index, gsplat from git (built against torch)
"${UVP[@]}" --no-build-isolation --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html -e '.[inference]'
./patching/hydra
# trimesh for exporting meshes (SAM 3 masks run in the vlab venv, which has a newer transformers)
"${UVP[@]}" trimesh
python -c "import torch, pytorch3d, kaolin, gsplat; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# checkpoints (gated)
if [ ! -f checkpoints/hf/pipeline.yaml ]; then
  if python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("facebook/sam-3d-objects", "checkpoints/pipeline.yaml", local_dir="/tmp/sam3d-probe")
PY
  then
    hf download --repo-type model --local-dir checkpoints/hf-download --max-workers 1 facebook/sam-3d-objects
    mv checkpoints/hf-download/checkpoints checkpoints/hf && rm -rf checkpoints/hf-download
  else
    echo "!! cannot read facebook/sam-3d-objects yet: request access on Hugging Face and log in, then re-run"
  fi
fi
echo "SAM 3D env ready: $MF/envs/sam3d-objects"
