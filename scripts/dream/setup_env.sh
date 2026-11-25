#!/usr/bin/env bash
# One env for the style-dream pipeline: vlab + Newton (the skild-ai fork the Steel examples pin) + SolverSteel
# + CUDA torch, on a Linux GPU machine. Kept apart from uv.lock: Newton's fork needs numpy >= 2.3.
#
#   scripts/dream/setup_env.sh            # -> .venv-steel/
#   .venv-steel/bin/vlab run style_dream_toy
set -euo pipefail
cd "$(dirname "$0")/../.."
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
NEWTON="newton[examples] @ git+https://github.com/skild-ai/newton.git@f6fc82a8b06d036e1e6d02a61d77a2d300476e8d"
VENV=${VENV:-.venv-steel}
[ -x "$VENV/bin/python" ] || uv venv -p 3.12 "$VENV"
PY="$VENV/bin/python"
uv pip install --python "$PY" "$NEWTON" "warp-lang==1.18.0"
uv pip install --python "$PY" torch --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
uv pip install --python "$PY" omegaconf pyyaml safetensors einops tqdm pillow imageio imageio-ffmpeg gymnasium \
  transformers huggingface-hub scipy trimesh matplotlib pytest
uv pip install --python "$PY" --no-deps -e .
"$PY" -c "import newton, warp, torch; print('newton', newton.__version__, 'warp', warp.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
