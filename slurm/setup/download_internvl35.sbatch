#!/bin/bash
# Download the original InternVL3.5 checkpoint from the HF mirror.
# This is an I/O job; it must not run on the login/debug node.
# Usage: sbatch slurm/setup/download_internvl35.sbatch
# Override MODEL_ROOT or HF_ENDPOINT with --export=ALL,... when needed.
#SBATCH --job-name=download-internvl35
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/setup/%x-%j.out
#SBATCH --error=logs/setup/%x-%j.out

set -euo pipefail
trap 'echo "DOWNLOAD FAILED at line $LINENO (exit $?)" >&2' ERR
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

MODEL_ROOT=${MODEL_ROOT:?MODEL_ROOT must be set}
MODEL_ID="${MODEL_ID:-OpenGVLab/InternVL3_5-8B}"
TARGET="$MODEL_ROOT/InternVL3_5-8B"
mkdir -p logs/setup "$MODEL_ROOT"

echo "model=$MODEL_ID"
echo "endpoint=$HF_ENDPOINT"
echo "target=$TARGET"

MODEL_ID="$MODEL_ID" TARGET="$TARGET" "$HF_PYTHON" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = os.environ["MODEL_ID"]
target = Path(os.environ["TARGET"])
target.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=repo_id,
    local_dir=target,
    max_workers=4,
)
print(f"downloaded={target}")
PY

test -f "$TARGET/config.json"
test -f "$TARGET/model.safetensors.index.json" || test -f "$TARGET/pytorch_model.bin.index.json"
printf 'download_complete model=%s target=%s endpoint=%s\n' "$MODEL_ID" "$TARGET" "$HF_ENDPOINT"
