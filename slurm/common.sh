#!/bin/bash
# Shared, overridable paths for Slurm entry points.

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SLURM_DIR/.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-/opt/${USER:?}/models}"
DATA_ROOT="${DATA_ROOT:-/opt/${USER:?}/datasets}"
VLLM_PYTHON="${VLLM_PYTHON:-$HOME/.venvs/vllm-0.25.1/bin/python}"
HF_PYTHON="${HF_PYTHON:-$HOME/.venvs/vlm-transformers/bin/python}"

export PROJECT_ROOT MODEL_ROOT DATA_ROOT VLLM_PYTHON HF_PYTHON
