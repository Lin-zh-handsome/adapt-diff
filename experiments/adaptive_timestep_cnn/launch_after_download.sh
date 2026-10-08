#!/usr/bin/env bash
set -euo pipefail
project=/home/hanjinwei/p1/project/diffusion_policy
experiment="$project/experiments/adaptive_timestep_cnn"
runs="$experiment/runs"
data="$experiment/data/can/ph/image.hdf5"
python=/home/hanjinwei/miniconda3/envs/robodiff/bin/python
status="$runs/formal_launch.log"
printf '%s waiting_for_official_image\n' "$(date -Is)" >> "$status"
while [ ! -f "$runs/download.exit" ]; do sleep 30; done
if [ "$(cat "$runs/download.exit")" != 0 ]; then
  printf '%s download_failed\n' "$(date -Is)" >> "$status"
  exit 1
fi
actual=$(stat -c %s "$data")
if [ "$actual" -ne 2010694944 ]; then
  printf '%s incomplete_size=%s\n' "$(date -Is)" "$actual" >> "$status"
  exit 1
fi
"$python" -c 'import h5py,sys; f=h5py.File(sys.argv[1],"r"); assert len(f["data"])>0; print("HDF5 readable: demos="+str(len(f["data"]))); f.close()' "$data" >> "$status" 2>&1
vanilla="$runs/vanilla/can_seed42"
adaptive="$runs/adaptive/can_seed42"
if [ -e "$vanilla" ] || [ -e "$adaptive" ]; then
  printf '%s existing_output_prevents_new_run\n' "$(date -Is)" >> "$status"
  exit 1
fi
cd "$project"
mkdir -p "$vanilla"
printf 'mode=vanilla\ntask=can_image\nrun_name=can_seed42\nseed=42\nphysical_gpu=0\nvisible_device=cuda:0\ndataset=%s\n' "$data" > "$vanilla/launch_config.txt"
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 nohup "$python" -u -m experiments.adaptive_timestep_cnn.train --mode vanilla --task can_image --run-name can_seed42 --device cuda:0 > "$vanilla/stdout.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$vanilla/train.pid"
printf '%s launched_vanilla gpu=0 pid=%s\n' "$(date -Is)" "$pid" >> "$status"
nohup bash "$experiment/queue_adaptive_after_vanilla.sh" > "$runs/adaptive_queue.stdout.log" 2>&1 < /dev/null &
printf '%s queued_adaptive supervisor_pid=%s\n' "$(date -Is)" "$!" >> "$status"

