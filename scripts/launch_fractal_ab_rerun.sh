#!/usr/bin/env bash
set -euo pipefail

project_directory=/home/boni/projects/FractalTokenizer-v2-rate-distortion/FractalTokenizer-v2-rate-distortion
environment_file=/home/boni/ai/envs/dgx-dl/bin/activate
session_name=fractal-imagenet100-ab-rerun
log_file="$project_directory/runs/imagenet100_fractal_ab_preferred/console.log"

mkdir -p "$(dirname "$log_file")"
tmux new-session -d -s "$session_name" "bash -lc \"set -o pipefail; source '$environment_file' && cd '$project_directory' && python scripts/train_fractal_ab_rerun.py 2>&1 | tee -a '$log_file'\""
echo "Started tmux session $session_name"
echo "Log: $log_file"
