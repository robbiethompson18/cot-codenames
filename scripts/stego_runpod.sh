#!/bin/bash
# Launch one stego training run on its own fresh RunPod pod. The pod uploads the run to Hugging Face and then deletes
# itself, so nothing keeps billing if this machine goes to sleep.
#
#   scripts/stego_runpod.sh <run-name> [train.py args...]     e.g.  scripts/stego_runpod.sh fixed8-strict --pool-size 8
#   STEGO_POD=<pod-id> scripts/stego_runpod.sh <run-name> ... reuse an existing pod (it is still deleted at the end)
#
# Needs HF_TOKEN and ANTHROPIC_API_KEY in the environment (source .envrc.local) and a clean git tree for the commit stamp.
set -euo pipefail
run=$1
shift
repo=robbiethompson2018/cot-codenames-stego
key=$HOME/.runpod/ssh/RunPod-Key-Go
sha=$(git rev-parse HEAD)

pod=${STEGO_POD:-}
if [ -z "$pod" ]; then
  # One H200 (141 GB) or B200 (180 GB) holds the 27B plus training activations at --micro-batch 4.
  for gpu in "NVIDIA H200" "NVIDIA B200"; do
    pod=$(runpodctl pod create --name "stego-$run" --gpu-id "$gpu" --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
      --container-disk-in-gb 160 --ports 22/tcp 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))' || true)
    [ -n "$pod" ] && break
  done
  [ -n "$pod" ] || { echo "$run: no H200 or B200 available"; exit 1; }
fi
echo "$run: pod $pod"

until info=$(runpodctl ssh info "$pod" 2>/dev/null) && echo "$info" | grep -q '"port"'; do sleep 10; done
read -r host port < <(echo "$info" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["ip"], d["port"])')
ssh_opts=(-i "$key" -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
until ssh "${ssh_opts[@]}" "root@$host" true 2>/dev/null; do sleep 5; done

rsync -az -e "ssh ${ssh_opts[*]}" --exclude .venv --exclude .git --exclude runs --exclude __pycache__ --exclude .ruff_cache \
  --exclude .envrc.local ./ "root@$host:/workspace/cot-codenames/"
# Keys go into a root-only file on the pod, never into the pod's RunPod config.
printf 'export HF_TOKEN=%s\nexport ANTHROPIC_API_KEY=%s\nexport HF_HOME=/root/hf\nexport STEGO_GIT_SHA=%s\nexport PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n' \
  "$HF_TOKEN" "$ANTHROPIC_API_KEY" "$sha" |
  ssh "${ssh_opts[@]}" "root@$host" 'umask 077; cat > /root/.stego_env'

args=$(printf '%q ' "$@")
# STEGO_SETUP=pip uses the image's own torch; `uv sync` re-downloads torch and has stalled for 30+ minutes in some
# datacenters. Either way \$py is the interpreter the run uses.
if [ "${STEGO_SETUP:-uv}" = pip ]; then
  setup='pip install -q --break-system-packages transformers peft accelerate huggingface_hub anthropic flash-linear-attention && pip install -q --break-system-packages --no-deps -e . ; py=python'
else
  setup='pip install -q uv 2>/dev/null; uv sync -q --group train; py="uv run python"'
fi
ssh "${ssh_opts[@]}" "root@$host" "cat > /workspace/run_$run.sh" <<REMOTE
source /root/.stego_env
cd /workspace/cot-codenames
$setup
\$py -m cot_codenames.stego.train --backend local --run $run --hf-repo $repo $args 2>&1 | tee /workspace/$run.log
# Whether training finished or crashed: keep the log with the run, upload, and release the pod.
mkdir -p runs/stego/$run && cp /workspace/$run.log runs/stego/$run/train.log
\$py -c "from huggingface_hub import upload_folder; upload_folder(repo_id='$repo', folder_path='runs/stego/$run', path_in_repo='runs/$run')"
runpodctl remove pod $pod || runpodctl stop pod $pod
REMOTE
ssh "${ssh_opts[@]}" "root@$host" "tmux kill-server 2>/dev/null; pkill -f '[u]v sync' 2>/dev/null; tmux new -d -s $run 'bash /workspace/run_$run.sh'"
echo "$run: started on $pod ($host:$port)"
