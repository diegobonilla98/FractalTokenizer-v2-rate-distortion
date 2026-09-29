#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/boni/projects/FractalTokenizer-v2-rate-distortion/FractalTokenizer-v2-rate-distortion
ENVIRONMENT=/home/boni/ai/envs/dgx-dl/bin/activate

source "$ENVIRONMENT"
cd "$PROJECT_ROOT"
export TORCH_CUDA_ARCH_LIST=12.1
python scripts/build_cuda_extension.py
