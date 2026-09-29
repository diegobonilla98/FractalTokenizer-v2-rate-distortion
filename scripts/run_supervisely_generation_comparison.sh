#!/usr/bin/env bash
set -euo pipefail

project_directory=/home/boni/projects/FractalTokenizer-v2-rate-distortion/FractalTokenizer-v2-rate-distortion
environment_file=/home/boni/ai/envs/dgx-dl/bin/activate
telegram_directory=/home/boni/ai/apps/TelegramBot
telegram_environment_file=/home/boni/.bashrc
log_file="$project_directory/runs/supervisely_generation/comparison/console.log"

telegram_exports=$(grep -E '^export TELEGRAM_(BOT_TOKEN|CHAT_ID)=' "$telegram_environment_file")
eval "$telegram_exports"
if [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]]; then
    echo "Telegram credentials were not found in $telegram_environment_file"
    exit 1
fi

source "$environment_file"
export PYTHONPATH="$telegram_directory:${PYTHONPATH:-}"
cd "$project_directory"
python scripts/train_supervisely_generation_comparison.py 2>&1 | tee -a "$log_file"
