#!/usr/bin/env bash
# UMI SLAM toolchain: the UMI repo, a small uv venv for its pipeline scripts, and the two docker images
# (ORB-SLAM3, OpenICC for GoPro IMU) its pipeline calls.
#   scripts/real2sim/setup_umi.sh [ROOT]        (default ROOT=~/real2sim)
set -euo pipefail
ROOT=${1:-$HOME/real2sim}
UMI_REV=${UMI_REV:-main}
mkdir -p "$ROOT" && cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"
[ -d umi ] || git clone -q https://github.com/real-stanford/universal_manipulation_interface umi
git -C umi checkout -q "$UMI_REV"
cd umi
# The UMI conda env is a full training env; the SLAM pipeline (00-07) needs only these.
uv venv -q --python 3.10 .venv
uv pip install -q --python .venv/bin/python "numpy<2" "opencv-contrib-python==4.7.0.72" "av<13" "PyExifTool==0.5.6" \
  pandas scipy click tqdm numba "zarr==2.16.1" "numcodecs==0.11.0" "imagecodecs==2023.9.18" scikit-fda dill matplotlib imageio imageio-ffmpeg pyyaml
sudo docker pull -q chicheng/orb_slam3:latest
sudo docker pull -q chicheng/openicc:latest
echo "UMI toolchain ready in $ROOT/umi"
