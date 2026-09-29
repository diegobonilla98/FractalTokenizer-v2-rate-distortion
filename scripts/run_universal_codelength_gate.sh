#!/usr/bin/env bash
set -euo pipefail

project_root=/home/boni/projects/FractalTokenizer-v2-rate-distortion/FractalTokenizer-v2-rate-distortion
environment_file=/home/boni/ai/envs/dgx-dl/bin/activate
telegram_directory=/home/boni/ai/apps/TelegramBot
telegram_environment_file=/home/boni/.bashrc
session_name=fractal_universal_codelength_gate
output_root=/mnt/hdd/fractaltokenizer_constant_information/universal_codelength_gate
log_file="$output_root/evaluation.log"

if [[ "${FRACTAL_UNIVERSAL_GATE_IN_TMUX:-0}" != "1" ]]; then
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "Session $session_name is already running"
        echo "Attach: tmux attach -t $session_name"
        exit 1
    fi
    mkdir -p "$output_root"
    tmux new-session -d -s "$session_name" \
        "cd '$project_root' && FRACTAL_UNIVERSAL_GATE_IN_TMUX=1 bash scripts/run_universal_codelength_gate.sh"
    echo "Started $session_name"
    echo "Attach: tmux attach -t $session_name"
    echo "Log: tail -f $log_file"
    exit 0
fi

telegram_exports=$(grep -E '^export TELEGRAM_(BOT_TOKEN|CHAT_ID)=' "$telegram_environment_file")
eval "$telegram_exports"
if [[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]]; then
    echo "Telegram credentials were not found in $telegram_environment_file"
    exit 1
fi

source "$environment_file"
export PYTHONPATH="$telegram_directory:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST=12.1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
cd "$project_root"
mkdir -p "$output_root"
python scripts/evaluate_universal_codelength_gate.py 2>&1 | tee -a "$log_file"
