#!/usr/bin/env bash
# Download a UMI session (raw GoPro MP4s) and run UMI's SLAM pipeline on it (steps 00-06).
#   scripts/real2sim/fetch_umi_session.sh [SESSION_DIR] [URL]
# Default: the UMI authors' example session (cup on saucer, 2.7K fisheye GoPro, ~730 MB),
#   https://real.stanford.edu/umi/data/example_demo_session/
# A Hugging Face folder works too: hf://datasets/<repo>/<folder> with mapping.mp4, gripper_calibration.mp4 and
# numbered demos (Data Scaling Laws' raw videos, e.g. .../pick_place_mouse/env_17/object_1); MAX_DEMOS (default 24)
# caps how many demos are fetched.
set -euo pipefail
SESSION=${1:-$HOME/real2sim/data/umi_cup_example}
URL=${2:-https://real.stanford.edu/umi/data/example_demo_session/}
UMI=${UMI_DIR:-$HOME/real2sim/umi}
MAX_DEMOS=${MAX_DEMOS:-24}
mkdir -p "$SESSION"
if ! ls "$SESSION"/raw_videos >/dev/null 2>&1 && ! ls "$SESSION"/*.MP4 "$SESSION"/*.mp4 >/dev/null 2>&1; then
  if [[ "$URL" == hf://datasets/* ]]; then
    rest=${URL#hf://datasets/}; repo=$(echo "$rest" | cut -d/ -f1-2); folder=$(echo "$rest" | cut -d/ -f3-)
    base="https://huggingface.co/datasets/$repo/resolve/main/$folder"
    names=$(curl -sL "https://huggingface.co/api/datasets/$repo/tree/main/$folder" | python3 -c "
import json, sys
names = [x['path'].rsplit('/', 1)[-1] for x in json.load(sys.stdin) if x['type'] == 'file']
special = [n for n in names if n.startswith(('mapping', 'gripper_cal'))]
demos = sorted((n for n in names if n.split('.')[0].isdigit()), key=lambda n: int(n.split('.')[0]))
print(' '.join(special + demos[:$MAX_DEMOS]))")
    for f in $names; do aria2c -q -x 8 -d "$SESSION" -o "$f" "$base/$f"; done
  else
    # the index is a plain file listing; fetch every MP4 it links
    for f in $(curl -sL "$URL" | grep -oE 'href="[^"]+\.(MP4|mp4)"' | sed 's/href="//;s/"$//' | sort -u); do
      aria2c -q -x 8 -d "$SESSION" -o "$(basename "$f")" "${URL%/}/$f"
    done
  fi
fi
ls -la "$SESSION"
cd "$UMI"
PY=.venv/bin/python
# run_slam_pipeline.py chains 00-06 but calls `python`; put the venv first on PATH.
export PATH="$UMI/.venv/bin:$PATH"
# docker needs root on a fresh VM unless the user's group change has taken effect
if ! docker info >/dev/null 2>&1; then
  mkdir -p "$HOME/.r2s-bin" && printf '#!/bin/sh\nexec sudo /usr/bin/docker "$@"\n' > "$HOME/.r2s-bin/docker"
  chmod +x "$HOME/.r2s-bin/docker"; export PATH="$HOME/.r2s-bin:$PATH"
fi
$PY run_slam_pipeline.py "$SESSION" || true
# UMI's gripper calibration wants the finger tags in 80 % of the calibration video's frames; in-the-wild sessions
# (Data Scaling Laws' env_17: 74 %) can fall just short. Retry with GRIPPER_TAG_THRESHOLD, then plan the dataset.
if [ ! -f "$SESSION/dataset_plan.pkl" ]; then
  for g in "$SESSION"/demos/gripper_calibration*; do
    [ -f "$g/gripper_range.json" ] || $PY scripts/calibrate_gripper_range.py --input "$g/tag_detection.pkl" \
      --output "$g/gripper_range.json" --tag_det_threshold "${GRIPPER_TAG_THRESHOLD:-0.7}"
  done
  $PY scripts_slam_pipeline/06_generate_dataset_plan.py --input "$SESSION"
fi
test -f "$SESSION/dataset_plan.pkl"
echo "SLAM done: $SESSION/dataset_plan.pkl"
