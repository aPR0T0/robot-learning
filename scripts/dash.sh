#!/usr/bin/env bash
# Live W&B-style dashboard of every running VM's vlab runs (same as `vlab dash`, without the vlab venv).
#   scripts/dash.sh                       # configs/dash/default.yaml, opens http://127.0.0.1:8765
#   scripts/dash.sh port=8800 'vms=[vla-lab-a100]'
# Needs an authenticated gcloud and python3 with pyyaml: .venv's if there is one, else the system python3,
# else uv fetches pyyaml into its cache. Nothing is installed on the VMs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -x .venv/bin/python ]]; then exec .venv/bin/python -m vlalab.dash "$@"; fi
if python3 -c 'import yaml' 2>/dev/null; then exec python3 -m vlalab.dash "$@"; fi
exec uv run --no-project --with pyyaml python -m vlalab.dash "$@"
