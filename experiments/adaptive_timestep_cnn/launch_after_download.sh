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
cd "$project"
for mode in vanilla adaptive; do
  out="$runs/$mode/can_seed42"
  if [ -e "$out" ]; then
    printf '%s existing_output=%s\n' "$(date -Is)" "$out" >> "$status"
    exit 1
  fi
done
for mode in vanilla adaptive; do
  gpu=0
  if [ "$mode" = adaptive ]; then gpu=1; fi
  out="$runs/$mode/can_seed42"
  mkdir -p "$out"
  printf 'mode=%s\ntask=can_image\nrun_name=can_seed42\nseed=42\nphysical_gpu=%s\nvisible_device=cuda:0\ndataset=%s\n' "$mode" "$gpu" "$data" > "$out/launch_config.txt"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 nohup "$python" -u -m experiments.adaptive_timestep_cnn.train --mode "$mode" --task can_image --run-name can_seed42 --device cuda:0 > "$out/stdout.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$out/train.pid"
  printf '%s launched mode=%s physical_gpu=%s pid=%s log=%s\n' "$(date -Is)" "$mode" "$gpu" "$pid" "$out/stdout.log" >> "$status"
done

