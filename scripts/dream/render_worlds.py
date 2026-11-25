"""Render a grid of randomised worlds at rest (MP4 + USD): python scripts/dream/render_worlds.py OUT_DIR [N]."""
import sys
from pathlib import Path

from vlalab.dream.steel_env import EnvConfig
from vlalab.stages.dream import render_bank

out = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
print(render_bank(EnvConfig(worlds=n, seed=7), out, "cuda:0"))
