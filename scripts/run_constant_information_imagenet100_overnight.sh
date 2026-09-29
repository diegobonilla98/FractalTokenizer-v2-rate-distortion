#!/usr/bin/env bash
set -euo pipefail

project_directory=/home/boni/projects/FractalTokenizer-v2-rate-distortion/FractalTokenizer-v2-rate-distortion
environment_file=/home/boni/ai/envs/dgx-dl/bin/activate
telegram_directory=/home/boni/ai/apps/TelegramBot
telegram_environment_file=/home/boni/.bashrc
session_name=constant-information-imagenet100
output_directory=/mnt/hdd/fractaltokenizer_constant_information/imagenet100_overnight
log_file="$output_directory/console.log"
runner="$project_directory/scripts/run_constant_information_imagenet100_overnight.sh"

if [[ "${FRACTAL_INSIDE_TMUX:-0}" != "1" ]]; then
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "tmux session already exists: $session_name"
        exit 1
    fi
    mkdir -p "$output_directory"
    tmux new-session -d -s "$session_name" "FRACTAL_INSIDE_TMUX=1 bash '$runner'"
    echo "Started tmux session $session_name"
    echo "Attach: tmux attach -t $session_name"
    echo "Log: tail -f $log_file"
    exit 0
fi

telegram_exports=$(grep -E '^export TELEGRAM_(BOT_TOKEN|CHAT_ID)=' "$telegram_environment_file" || true)
eval "$telegram_exports"
if [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]]; then
    echo "Telegram credentials were not found in $telegram_environment_file"
    exit 1
fi

source "$environment_file"
export PYTHONPATH="$telegram_directory:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST=12.1
cd "$project_directory"
mkdir -p "$output_directory"
python scripts/build_cuda_extension.py
python scripts/train_constant_information_imagenet100.py 2>&1 | tee -a "$log_file"
