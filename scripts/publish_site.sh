#!/usr/bin/env bash
# Build the node map and copy it into the apr0t0.github.io checkout, where it is served at
# https://apr0t0.github.io/robot-learning/ (the site repo's own Pages workflow deploys it).
#
#   scripts/publish_site.sh [path/to/apr0t0.github.io] [--results runs/]
set -euo pipefail
cd "$(dirname "$0")/.."
SITE_REPO=${1:-../apr0t0.github.io}
shift || true
[[ -d "$SITE_REPO/.git" ]] || { echo "not a git checkout: $SITE_REPO"; exit 1; }
PY_SET=${PY:-}
PY=${PY:-python3}
[[ -z "${PY_SET:-}" && -x .venv/bin/python ]] && PY=.venv/bin/python
PYTHONPATH=src "$PY" scripts/build_site.py "$@"
mkdir -p "$SITE_REPO/robot-learning"
cp site/index.html site/style.css site/app.js site/graph.json site/results.json "$SITE_REPO/robot-learning/"
echo "copied to $SITE_REPO/robot-learning/ - commit and push the site repo to deploy"
