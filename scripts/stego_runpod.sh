#!/bin/bash
# Launch a stego training run on a fresh RunPod pod. UNTESTED end to end as of 2026-10-08 (the account had no balance).
#
#   scripts/stego_runpod.sh up                      -> create the pod, print its id
#   scripts/stego_runpod.sh run <pod-id> <args...>  -> sync the repo and start train.py in tmux with <args...>
#   scripts/stego_runpod.sh down <pod-id>           -> delete the pod (billing stops)
#
# Both models in bf16 need about 75 GB (Qwen3.6-27B 56 GB + Qwen3.5-9B 19 GB), so an 80 GB card leaves no room to
# train. An H200 (141 GB) fits them with --micro-batch 2.
set -euo pipefail

case "$1" in
up)
  runpodctl pod create --name cot-stego --gpu-id "NVIDIA H200" --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
    --container-disk-in-gb 40 --volume-in-gb 150 --ports 22/tcp
  ;;
run)
  pod=$2
  shift 2
  # ASSUMPTION: `runpodctl ssh info` returns JSON with "ip" and "port"; check its real output on first use.
  read -r host port < <(runpodctl ssh info "$pod" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["ip"], d["port"])')
  rsync -az -e "ssh -p $port" --exclude .venv --exclude .git --exclude runs ./ "root@$host:/workspace/cot-codenames/"
  # Keys are passed on the ssh command line so they are never stored in the pod's RunPod config.
  ssh -p "$port" "root@$host" "cd /workspace/cot-codenames && pip install -q uv && \
    tmux new -d -s stego 'HF_HOME=/workspace/hf HF_TOKEN=$HF_TOKEN ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
    uv run --group train python -m cot_codenames.stego.train --backend local --micro-batch 2 \
    --hf-repo robbiethompson2018/cot-codenames-stego $* 2>&1 | tee /workspace/stego.log'"
  echo "started; tail with: ssh -p $port root@$host tail -f /workspace/stego.log"
  ;;
down)
  runpodctl pod delete "$2"
  ;;
esac
