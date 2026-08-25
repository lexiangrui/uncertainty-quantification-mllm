#!/bin/bash
# Train the LoRA adapter for the original OpenGVLab InternVLChatModel.
# The model is downloaded on the login node; this job is compute-only and offline.
#SBATCH --job-name=internvl35-original-lora
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
# InternVL3.5-8B bf16 plus LoRA fits within the per-GPU 64G host budget.
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/lora/%x-%j.out
#SBATCH --error=logs/lora/%x-%j.err

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/lexiangrui/Uncertainty-Quantification-of-MLLM}
source "$PROJECT_ROOT/slurm/common.sh"
PYTHON=${PYTHON:-$HF_PYTHON}
CONFIG=${CONFIG:-$PROJECT_ROOT/LoRA/configs/internvl3_5_original_lora.json}

cd "$PROJECT_ROOT"
mkdir -p logs/lora
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT/LoRA/src:$PROJECT_ROOT:${PYTHONPATH:-}"

exec "$PYTHON" LoRA/scripts/train_multimodal_lora.py \
  --config "$CONFIG" \
  "$@"
