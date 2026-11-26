#!/usr/bin/env bash
# PPO from cameras on the UMI mouse scene (docs/pixel_ppo.md): the curriculum run (privileged state + cameras ->
# cameras + joints), then every PPO variant evaluated on the same worlds, then the plots. Needs a finished
# umi_style_dream run (worlds, style, dreamer, policy_rl) and the style-dream env.
#
#   scripts/dream/run_pixel.sh                       # VLAB=.venv-steel/bin/vlab by default
#   FLAT=1 scripts/dream/run_pixel.sh                # also the no-curriculum run, side by side (add
#                                                    # policy_pixel_flat.policy to eval_pixel's pixel_policies)
#
# The curriculum run keeps its frames on the GPU (image_store auto); the no-curriculum one in pinned host memory.
set -euo pipefail
cd "$(dirname "$0")/../.."
VLAB=${VLAB:-.venv-steel/bin/vlab}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/logs
"$VLAB" run umi_style_dream --only policy_pixel "$@" > runs/logs/policy_pixel.log 2>&1 &
p1=$!
if [[ "${FLAT:-0}" == 1 ]]; then
  sleep 240  # let the first run claim its GPU frame buffer before the second one starts
  "$VLAB" run umi_style_dream --only policy_pixel_flat policy_pixel_flat.agent.image_store=cpu "$@" \
    > runs/logs/policy_pixel_flat.log 2>&1 &
  wait $!
fi
wait $p1
"$VLAB" run umi_style_dream --only eval_pixel > runs/logs/eval_pixel.log 2>&1
"$(dirname "$VLAB")/python" scripts/dream/plot_pixel_ppo.py runs/umi_style_dream docs/images/pixel_ppo
