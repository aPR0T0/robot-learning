#!/usr/bin/env bash
# The whole umi_style_dream run on one GPU machine: real2sim's nodes in the real2sim vlab venv, the style-dream
# nodes in .venv-steel (scripts/dream/setup_env.sh), the three policies concurrently, then the evaluation.
#   scripts/dream/run_umi.sh [run_dir] [extra overrides...]
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=${1:-runs/umi_style_dream}; shift || true
R2S=${R2S_VLAB:-.venv/bin/vlab}
VLAB=${VLAB:-.venv-steel/bin/vlab}
$R2S run umi_style_dream --run-dir "$RUN" --only demo,masks,scene,physics,sim "$@"
$VLAB run umi_style_dream --run-dir "$RUN" --only style,worlds,dreamer "$@"
pids=()
for n in policy policy_rl policy_dream; do
  $VLAB run umi_style_dream --run-dir "$RUN" --only $n "$@" > "$RUN/$n.log" 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
$VLAB run umi_style_dream --run-dir "$RUN" --only eval "$@"
