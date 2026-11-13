#!/usr/bin/env bash
# Create / use / stop a GCP GPU VM for this repo.
#
#   scripts/gcp_vm.sh create        # 1x L4 (g2-standard-8), Ubuntu 24.04 + NVIDIA 580 driver image
#   scripts/gcp_vm.sh push          # copy this checkout to the VM (no git auth needed)
#   scripts/gcp_vm.sh setup         # push + ./setup.sh on the VM
#   scripts/gcp_vm.sh run reach_smoke [overrides...]   # run a pipeline on the VM (in tmux-free nohup)
#   scripts/gcp_vm.sh pull          # copy runs/ back
#   scripts/gcp_vm.sh ssh | stop | start | delete
#
# Settings (env vars): PROJECT, ZONE, VM, MACHINE, GPU, DISK_GB, SPOT=1
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}
ZONE=${ZONE:-us-central1-a}
VM=${VM:-vla-lab-l4}
MACHINE=${MACHINE:-g2-standard-8}
GPU=${GPU:-nvidia-l4}
DISK_GB=${DISK_GB:-200}
G=(--project "$PROJECT" --zone "$ZONE")

cmd=${1:-help}; shift || true
case "$cmd" in
  create)
    extra=()
    [[ "${SPOT:-0}" == 1 ]] && extra+=(--provisioning-model=SPOT --instance-termination-action=STOP)
    gcloud compute instances create "$VM" "${G[@]}" --machine-type "$MACHINE" \
      --accelerator "type=$GPU,count=1" --maintenance-policy TERMINATE \
      --image-family ubuntu-accelerator-2404-amd64-with-nvidia-580 --image-project ubuntu-os-accelerator-images \
      --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-balanced --labels purpose=vla-lab "${extra[@]}"
    ;;
  push)
    tar --exclude=.venv --exclude=runs --exclude=.git --exclude='__pycache__' --exclude=site/node_modules \
      -czf /tmp/robot-learning.tgz .
    gcloud compute scp "${G[@]}" /tmp/robot-learning.tgz "$VM":/tmp/robot-learning.tgz
    gcloud compute ssh "$VM" "${G[@]}" --command 'mkdir -p ~/robot-learning && tar -xzf /tmp/robot-learning.tgz -C ~/robot-learning'
    ;;
  setup)
    "$0" push
    gcloud compute ssh "$VM" "${G[@]}" --command 'cd ~/robot-learning && ./setup.sh'
    ;;
  run)
    p=${1:?pipeline}; shift
    gcloud compute ssh "$VM" "${G[@]}" --command "cd ~/robot-learning && source .venv/bin/activate && \
      nohup vlab run $p $* > runs_$p.log 2>&1 & echo started; echo 'tail -f ~/robot-learning/runs_$p.log'"
    ;;
  pull)
    mkdir -p runs
    gcloud compute scp --recurse "${G[@]}" "$VM":~/robot-learning/runs/. runs/
    ;;
  ssh) gcloud compute ssh "$VM" "${G[@]}" "$@" ;;
  stop) gcloud compute instances stop "$VM" "${G[@]}" ;;
  start) gcloud compute instances start "$VM" "${G[@]}" ;;
  delete) gcloud compute instances delete "$VM" "${G[@]}" ;;
  *) sed -n '2,13p' "$0" ;;
esac
