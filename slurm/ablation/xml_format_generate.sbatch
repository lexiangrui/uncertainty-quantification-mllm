#!/bin/bash
#SBATCH --job-name=xml-ablation
#SBATCH --partition=batch,debug01
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/ablation/%x-%A_%a.out
#SBATCH --error=logs/ablation/%x-%A_%a.out

set -euo pipefail
trap 'echo "SCRIPT FAILED at line $LINENO (exit $?)" >&2' ERR
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"

MODELS=(llava qwen internvl)
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}]}"

cd "$PROJECT_ROOT"
mkdir -p logs/ablation
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PATH="$(dirname "$HF_PYTHON"):$PATH"

"$HF_PYTHON" scripts/ablation/run_xml_format_generation.py \
  --model "$MODEL" \
  --model-root "$MODEL_ROOT" \
  --data-root "$DATA_ROOT"
