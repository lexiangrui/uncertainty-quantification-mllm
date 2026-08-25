#!/bin/bash
# Shared, overridable paths for Slurm entry points.

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SLURM_DIR/.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-/opt/${USER:?}/models}"
DATA_ROOT="${DATA_ROOT:-/opt/${USER:?}/datasets}"
HF_PYTHON="${HF_PYTHON:-$HOME/.venvs/MLLM-UQ/bin/python}"
if [[ ! -x "$HF_PYTHON" ]]; then
  echo "MLLM-UQ Python is missing or not executable: $HF_PYTHON" >&2
  return 1 2>/dev/null || exit 1
fi

# torch/FlashAttention JIT compilation may include <Python.h>.  The compute
# nodes provide the Python 3.12 runtime but not the matching system dev package;
# use preinstalled user-local headers when available.
PYTHON312_DEV="${PYTHON312_DEV:-$HOME/.local/python312-dev}"
if [[ -f "$PYTHON312_DEV/usr/include/python3.12/Python.h" ]]; then
  export C_INCLUDE_PATH="$PYTHON312_DEV/usr/include/python3.12:$PYTHON312_DEV/usr/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
  export CPLUS_INCLUDE_PATH="$PYTHON312_DEV/usr/include/python3.12:$PYTHON312_DEV/usr/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
  export LIBRARY_PATH="$PYTHON312_DEV/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi

export PROJECT_ROOT MODEL_ROOT DATA_ROOT HF_PYTHON
