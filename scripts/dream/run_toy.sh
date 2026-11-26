#!/usr/bin/env bash
# Full toy run on one GPU: upstream nodes, then the three policies concurrently (they only share the cache),
# then the evaluation. Usage: scripts/dream/run_toy.sh [run_dir] [extra overrides...]
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=${1:-runs/style_dream_toy}; shift || true
VLAB=${VLAB:-.venv-steel/bin/vlab}
$VLAB run style_dream_toy --run-dir "$RUN" --only demo,style,worlds,dreamer "$@"
pids=()
for n in policy policy_rl policy_dream; do
  $VLAB run style_dream_toy --run-dir "$RUN" --only $n "$@" > "$RUN/$n.log" 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
$VLAB run style_dream_toy --run-dir "$RUN" --only eval "$@"
