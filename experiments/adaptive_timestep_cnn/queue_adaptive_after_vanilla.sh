#!/usr/bin/env bash
set -euo pipefail
project=/home/hanjinwei/p1/project/diffusion_policy
root="$project/experiments/adaptive_timestep_cnn"
runs="$root/runs"
status="$runs/adaptive_queue.log"
vanilla="$runs/vanilla/can_seed42"
adaptive="$runs/adaptive/can_seed42"
python=/home/hanjinwei/miniconda3/envs/robodiff/bin/python
pid=$(cat "$vanilla/train.pid")
printf '%s waiting_for_vanilla pid=%s\n' "$(date -Is)" "$pid" >> "$status"
while ps -p "$pid" -o args= 2>/dev/null | grep -Fq 'experiments.adaptive_timestep_cnn.train --mode vanilla'; do
  sleep 120
done
"$python" -c 'import json,pathlib,sys; out=pathlib.Path(sys.argv[1]); rows=(out/"logs.jsonl").read_text().splitlines(); assert rows and json.loads(rows[-1])["epoch"]==7999; assert (out/"latest.ckpt").is_file()' "$vanilla" >> "$status" 2>&1 || {
  printf '%s vanilla_incomplete_adaptive_not_started\n' "$(date -Is)" >> "$status"
  exit 1
}
if [ -e "$adaptive" ]; then
  printf '%s existing_adaptive_output=%s\n' "$(date -Is)" "$adaptive" >> "$status"
  exit 1
fi
while true; do
  free0=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  free1=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if [ "$free1" -ge 16000 ]; then gpu=1; break; fi
  if [ "$free0" -ge 16000 ]; then gpu=0; break; fi
  printf '%s waiting_for_free_gpu gpu0=%s gpu1=%s\n' "$(date -Is)" "$free0" "$free1" >> "$status"
  sleep 300
done
mkdir -p "$adaptive"
printf 'mode=adaptive\ntask=can_image\nrun_name=can_seed42\nseed=42\nphysical_gpu=%s\nvisible_device=cuda:0\ndataset=%s\n' "$gpu" "$root/data/can/ph/image.hdf5" > "$adaptive/launch_config.txt"
cd "$project"
CUDA_VISIBLE_DEVICES="$gpu" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 nohup "$python" -u -m experiments.adaptive_timestep_cnn.train --mode adaptive --task can_image --run-name can_seed42 --device cuda:0 > "$adaptive/stdout.log" 2>&1 < /dev/null &
adaptive_pid=$!
printf '%s\n' "$adaptive_pid" > "$adaptive/train.pid"
printf '%s launched_adaptive physical_gpu=%s pid=%s log=%s\n' "$(date -Is)" "$gpu" "$adaptive_pid" "$adaptive/stdout.log" >> "$status"

