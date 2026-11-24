#!/usr/bin/env bash
# Newton + Skild AI's Steel solver: the steel repo's examples project, pinned to its Newton revision.
#   scripts/real2sim/setup_steel.sh [ROOT]      (default ROOT=~/real2sim)
set -euo pipefail
ROOT=${1:-$HOME/real2sim}
STEEL_REV=${STEEL_REV:-main}
mkdir -p "$ROOT" && cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
[ -d steel ] || git clone -q https://github.com/skild-ai/steel steel
git -C steel checkout -q "$STEEL_REV"
cd steel/examples
uv sync --python 3.12   # imgui-bundle (Newton viewer) has no cp310 wheel
# trimesh + fast-simplification (SAM 3D meshes have ~1M triangles) + coacd for collision hulls, imageio for videos
uv pip install -q --python .venv/bin/python trimesh "coacd>=1.0.5" fast-simplification imageio imageio-ffmpeg scipy pyyaml
uv run --no-sync python -c "import newton, warp as wp; wp.init(); print('newton', newton.__version__, 'warp', wp.__version__, 'cuda', wp.is_cuda_available()); from newton.solvers import SolverSteel; print('SolverSteel OK')"
echo "Steel ready in $ROOT/steel/examples"
