#!/usr/bin/env bash
set -euo pipefail

project_root="$HOME/projects/FractalTokenizer-v2-rate-distortion/FractalTokenizer-v2-rate-distortion"
environment_path="$HOME/ai/envs/dgx-dl/bin/activate"

source "$environment_path"
cd "$project_root"
python scripts/gradio_fractal_ab.py
