#!/usr/bin/env bash
set -euo pipefail
root=/home/hanjinwei/p1/project/diffusion_policy/experiments/adaptive_timestep_cnn
mkdir -p "$root/data/can/ph" "$root/runs"
trap 'printf "%s\n" "$?" > "$root/runs/download.exit"' EXIT
curl --proxy http://127.0.0.1:7890 --fail --location --continue-at - --retry 8 --retry-delay 5 --output "$root/data/can/ph/image.hdf5"  https://downloads.cs.stanford.edu/downloads/rt_benchmark/can/ph/image.hdf5
