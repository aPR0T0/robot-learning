#!/usr/bin/env bash
# Download a UMI session (raw GoPro MP4s) and run UMI's SLAM pipeline on it (steps 00-06).
#   scripts/real2sim/fetch_umi_session.sh [SESSION_DIR] [URL]
# Default: the UMI authors' example session (cup on saucer, 2.7K fisheye GoPro, ~730 MB),
#   https://real.stanford.edu/umi/data/example_demo_session/
set -euo pipefail
SESSION=${1:-$HOME/real2sim/data/umi_cup_example}
URL=${2:-https://real.stanford.edu/umi/data/example_demo_session/}
UMI=${UMI_DIR:-$HOME/real2sim/umi}
mkdir -p "$SESSION"
if ! ls "$SESSION"/raw_videos >/dev/null 2>&1 && ! ls "$SESSION"/*.MP4 >/dev/null 2>&1; then
  # the index is a plain file listing; fetch every MP4 it links
  for f in $(curl -sL "$URL" | grep -oE 'href="[^"]+\.(MP4|mp4)"' | sed 's/href="//;s/"$//' | sort -u); do
    aria2c -q -x 8 -d "$SESSION" -o "$(basename "$f")" "${URL%/}/$f"
  done
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
$PY run_slam_pipeline.py "$SESSION"
echo "SLAM done: $SESSION/dataset_plan.pkl"
