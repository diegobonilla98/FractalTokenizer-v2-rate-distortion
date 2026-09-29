#!/usr/bin/env bash
set -euo pipefail

project_directory=/home/boni/projects/FractalTokenizer-v2-rate-distortion/FractalTokenizer-v2-rate-distortion
source_dataset_directory=/mnt/hdd/coco2017
staged_dataset_directory=/dev/shm/coco2017
session_name=fractal-coco-continuous-ae
log_file="$project_directory/runs/coco_continuous_autoencoder/console.log"
runner="$project_directory/scripts/run_coco_continuous_autoencoder.sh"

if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "tmux session already exists: $session_name"
    exit 1
fi

mkdir -p "$staged_dataset_directory/train2017" "$staged_dataset_directory/val2017" "$(dirname "$log_file")"
rsync -a --ignore-existing "$source_dataset_directory/train2017/" "$staged_dataset_directory/train2017/"
rsync -a --ignore-existing "$source_dataset_directory/val2017/" "$staged_dataset_directory/val2017/"
tmux new-session -d -s "$session_name" "bash '$runner'"
echo "Started tmux session $session_name"
echo "Log: $log_file"
echo "Attach: tmux attach -t $session_name"
