#!/bin/bash
# Launch one stego training run on its own RunPod pod. The pod uploads the run to Hugging Face and then deletes itself,
# so nothing keeps billing if this machine goes to sleep.
#
#   scripts/stego_runpod.sh <run-name> [train.py args...]     e.g.  scripts/stego_runpod.sh fixed8 --pool-size 8
#   STEGO_NOTE="one-line purpose" ...                         shown on the dashboard (cot_codenames.stego.dash)
#   STEGO_POD=<pod-id> ...                                    reuse an existing pod (it is still deleted at the end)
#   STEGO_KEEP=1 ...                                          leave the pod up afterwards, to queue more work on it
#   STEGO_DC=US-GA-2,CA-MTL-3 ...                             pin datacenters
#
# Needs HF_TOKEN and ANTHROPIC_API_KEY in the environment (source .envrc.local).
#
# The script only returns once the pod has a working environment: it installs the dependencies itself and checks that
# torch sees the GPU, and replaces the pod if that fails. On 2026-10-08/09 pods failed in every one of these ways: an
# NVIDIA driver too old for the wheel, `uv sync` stalling on a slow link, and environments that installed but could not
# train. When setup ran unattended inside the pod, those pods just vanished without a log.
set -uo pipefail
run=$1
shift
repo=robbiethompson2018/cot-codenames-stego
key=$HOME/.runpod/ssh/RunPod-Key-Go
sha=$(git rev-parse HEAD)
image=runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
pod=${STEGO_POD:-}

ready=""
for attempt in 1 2 3 4 5; do
  if [ -z "$pod" ]; then
    # One H200 (141 GB) or B200 (180 GB) holds the 27B plus training activations at --micro-batch 2.
    for gpu in "NVIDIA H200" "NVIDIA B200"; do
      pod=$(runpodctl pod create --name "stego-$run" --gpu-id "$gpu" --image $image --container-disk-in-gb 160 --ports 22/tcp \
        ${STEGO_DC:+--data-center-ids "$STEGO_DC"} 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))' 2>/dev/null)
      [ -n "$pod" ] && break
    done
    [ -n "$pod" ] || { echo "$run: no H200 or B200 available"; exit 1; }
  fi
  echo "$run: pod $pod (attempt $attempt)"
  until info=$(runpodctl ssh info "$pod" 2>/dev/null) && echo "$info" | grep -q '"port"'; do sleep 10; done
  read -r host port < <(echo "$info" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["ip"], d["port"])')
  ssh_opts=(-i "$key" -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=30)
  until ssh "${ssh_opts[@]}" "root@$host" true 2>/dev/null; do sleep 5; done

  # The lockfile pins torch's CUDA 12.6 build on Linux, which needs NVIDIA driver 560 or newer.
  driver=$(ssh "${ssh_opts[@]}" "root@$host" "nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1")
  problem=""
  [ "${driver:-0}" -ge 560 ] || problem="NVIDIA driver ${driver:-unknown} is older than 560"
  if [ -z "$problem" ]; then
    rsync -az -e "ssh ${ssh_opts[*]}" --exclude .venv --exclude .git --exclude runs --exclude __pycache__ --exclude .ruff_cache \
      --exclude .envrc.local ./ "root@$host:/workspace/cot-codenames/" || problem="rsync failed"
  fi
  if [ -z "$problem" ]; then
    # Do not swap in the pod image's own torch to save download time. Tried twice on 2026-10-08 and it failed three ways:
    # flash-linear-attention rejects the image's Triton for training, upgrading Triton pulls a torch that breaks the
    # image's torchaudio, and Triton 3.8 then finds no GPU driver.
    check='import torch, fla, peft, anthropic; assert torch.cuda.is_available(), "torch sees no GPU"'
    ssh "${ssh_opts[@]}" "root@$host" "cd /workspace/cot-codenames && pip install -q uv 2>/dev/null; \
      (timeout 900 uv sync -q --group train || timeout 900 uv sync -q --group train) && uv run python -c '$check'" \
      || problem="environment setup failed (uv sync or the GPU check)"
  fi
  if [ -z "$problem" ]; then
    ready=1
    break
  fi
  echo "$run: pod $pod unusable: $problem; replacing it"
  runpodctl pod delete "$pod" >/dev/null
  pod=""
done
[ -n "$ready" ] || { echo "$run: no usable pod after 5 tries"; exit 1; }

# Keys go into a root-only file on the pod, never into the pod's RunPod config.
printf 'export HF_TOKEN=%s\nexport ANTHROPIC_API_KEY=%s\nexport HF_HOME=/root/hf\nexport STEGO_GIT_SHA=%s\nexport PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n' \
  "$HF_TOKEN" "$ANTHROPIC_API_KEY" "$sha" |
  ssh "${ssh_opts[@]}" "root@$host" 'umask 077; cat > /root/.stego_env'

args=$(printf '%q ' "$@")
ssh "${ssh_opts[@]}" "root@$host" "cat > /workspace/run_$run.sh" <<REMOTE
source /root/.stego_env
cd /workspace/cot-codenames
uv run python -m cot_codenames.stego.train --backend local --run $run --hf-repo $repo $args 2>&1 | tee /workspace/$run.log
# Whether training finished or crashed: keep the log with the run, upload, and release the pod.
mkdir -p runs/stego/$run && cp /workspace/$run.log runs/stego/$run/train.log
uv run python -c "from huggingface_hub import upload_folder; upload_folder(repo_id='$repo', folder_path='runs/stego/$run', path_in_repo='runs/$run')"
# The pod's own API key lives in PID 1's environment, not in ssh or tmux shells; runpodctl needs it.
export RUNPOD_API_KEY=\$(tr '\\0' '\\n' < /proc/1/environ | grep ^RUNPOD_API_KEY= | cut -d= -f2-)
[ -n "${STEGO_KEEP:-}" ] || runpodctl remove pod $pod || runpodctl stop pod $pod
REMOTE
# The bracket keeps pkill from matching this very command line and killing the shell that runs it.
ssh "${ssh_opts[@]}" "root@$host" "tmux kill-server 2>/dev/null; pkill -f '[s]tego.train' 2>/dev/null; rm -f /workspace/$run.log; tmux new -d -s $run 'bash /workspace/run_$run.sh'"

# A local record of the run for the dashboard.
mkdir -p "runs/stego/$run"
python3 - "$run" "$pod" "$host" "$port" "$sha" "${STEGO_NOTE:-}" "$@" <<'PY' > "runs/stego/$run/launch.json"
import json, sys, time
run, pod, host, port, sha, note, *args = sys.argv[1:]
print(json.dumps({"run": run, "note": note, "pod": pod, "host": host, "port": int(port), "git_sha": sha, "args": args,
                  "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2))
PY
echo "$run: started on $pod ($host:$port)"
