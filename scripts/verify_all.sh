#!/usr/bin/env bash
# Run pipelines one after another (so timing / latency numbers don't contend for the GPU), keep going
# past failures, and print a pass/fail table. Logs go to runs/_logs/<pipeline>.log.
#
#   scripts/verify_all.sh                                   # the default set below
#   scripts/verify_all.sh reach_smoke pusht_lerobot         # just these
#   OVERRIDES="vla.train.steps=50" scripts/verify_all.sh reach_smoke
set -uo pipefail
cd "$(dirname "$0")/.."
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
PIPES=("$@")
((${#PIPES[@]})) || PIPES=(reach_smoke configs/pipeline/reach_heads.py mock_hardware pusht_lerobot reach_full aloha_lerobot)
mkdir -p runs/_logs
declare -A STATUS SECS
for p in "${PIPES[@]}"; do
  name=$(basename "${p%.*}")
  log=runs/_logs/$name.log
  echo "==> $name  (log: $log)"
  t0=$(date +%s)
  # shellcheck disable=SC2086
  if vlab run "$p" ${OVERRIDES:-} > "$log" 2>&1; then STATUS[$name]=ok; else STATUS[$name]=FAILED; fi
  SECS[$name]=$(( $(date +%s) - t0 ))
  echo "    ${STATUS[$name]} in ${SECS[$name]}s"
  [[ ${STATUS[$name]} == ok ]] || tail -5 "$log" | sed 's/^/    /'
done
echo
printf '%-16s %-7s %s\n' pipeline status seconds
fail=0
for p in "${PIPES[@]}"; do
  name=$(basename "${p%.*}")
  printf '%-16s %-7s %s\n' "$name" "${STATUS[$name]}" "${SECS[$name]}"
  [[ ${STATUS[$name]} == ok ]] || fail=1
done
exit $fail
