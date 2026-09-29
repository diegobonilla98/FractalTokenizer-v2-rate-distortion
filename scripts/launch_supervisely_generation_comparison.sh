#!/usr/bin/env bash
set -euo pipefail

project_directory=/home/boni/projects/FractalTokenizer-v2-rate-distortion/FractalTokenizer-v2-rate-distortion
source_image_directory=/mnt/hdd/supervisely_person_clean_2667_img/images
staged_image_directory=/dev/shm/supervisely_person_clean_2667_img/images
session_name=fractal-supervisely-generation
log_file="$project_directory/runs/supervisely_generation/comparison/console.log"
runner="$project_directory/scripts/run_supervisely_generation_comparison.sh"

if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "tmux session already exists: $session_name"
    exit 1
fi

mkdir -p "$staged_image_directory" "$(dirname "$log_file")"
rsync -a --ignore-existing "$source_image_directory/" "$staged_image_directory/"
tmux new-session -d -s "$session_name" "bash '$runner'"
echo "Started tmux session $session_name"
echo "Log: $log_file"
echo "Attach: tmux attach -t $session_name"
