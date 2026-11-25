#!/usr/bin/env bash
# Public object databases with known masses, for data/physics_match:
#   YCB (Calli et al. 2015): 77 objects, masses measured on a scale, scanned meshes (elpis-lab packaging, MIT)
#   ABO (Collins et al. 2022, CC BY 4.0): product listings with item weight, dimensions, material, catalog photos
#   scripts/real2sim/setup_physics_db.sh [ROOT]      (default ROOT=~/real2sim)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/common.sh"
ROOT=${1:-$HOME/real2sim}
YCB_REV=${YCB_REV:-$(pinned ycb_dataset)}   # default: the thirdparty/ submodule's commit
mkdir -p "$ROOT/data/abo" && cd "$ROOT/data"
clone_at ycb https://github.com/elpis-lab/ycb_dataset "$YCB_REV" --depth 1
cd abo
S3=https://amazon-berkeley-objects.s3.amazonaws.com/archives
[ -d listings ] || { aria2c -q -x 8 "$S3/abo-listings.tar" && tar -xf abo-listings.tar && rm abo-listings.tar; }
[ -d images ] || { aria2c -q -x 16 "$S3/abo-images-small.tar" && tar -xf abo-images-small.tar && rm abo-images-small.tar; }
curl -s -o LICENSE-CC-BY-4.0.txt https://amazon-berkeley-objects.s3.amazonaws.com/LICENSE-CC-BY-4.0.txt
echo "YCB: $(ls "$ROOT/data/ycb/ycb" | grep -c urdf) objects; ABO: $(ls listings/metadata | wc -l) listing files, images in $PWD/images"
# one catalog-style render per YCB object, so the classifier compares YCB and ABO entries photo to photo
export PATH="$HOME/.local/bin:$PATH"
cd "$ROOT"
[ -x render-venv/bin/python ] || { uv venv -q --python 3.11 render-venv && uv pip install -q --python render-venv/bin/python pyrender trimesh pillow numpy; }
PYOPENGL_PLATFORM=egl render-venv/bin/python "$HERE/render_ycb.py" "$ROOT/data/ycb"
