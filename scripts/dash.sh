#!/usr/bin/env bash
# The live dashboard of every running VM's vlab runs (dashboard/README.md); same as `vlab dash`.
#   scripts/dash.sh                       # configs/dash/default.yaml, opens http://127.0.0.1:8765
#   scripts/dash.sh port=8800 'vms=[vla-lab-a100]'
#   scripts/dash.sh status [--json]       # one report of every VM's GPU, processes and runs, then exit
# Needs an authenticated gcloud and python3 with pyyaml: .venv's if there is one, else the system python3,
# else uv fetches pyyaml into its cache. Nothing is installed on the VMs.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -x .venv/bin/python ]]; then exec .venv/bin/python dashboard/hub.py "$@"; fi
if python3 -c 'import yaml' 2>/dev/null; then exec python3 dashboard/hub.py "$@"; fi
exec uv run --no-project --with pyyaml python dashboard/hub.py "$@"
