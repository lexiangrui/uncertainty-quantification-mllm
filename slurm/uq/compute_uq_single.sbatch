#!/bin/bash
#SBATCH --job-name=uq
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/uq/%x-%j-%N.out
#SBATCH --error=logs/uq/%x-%j-%N.out

# Single unit UQ computation (PPL / SE / UMPIRE) for one (MODEL, DATASET) pair.
# Usage: sbatch --export=MODEL=llava,DATASET=vilp,RUN_ID=my-run slurm/uq/compute_uq_single.sbatch
set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"

MODEL=${MODEL:?MODEL must be llava|qwen|internvl}
DATASET=${DATASET:?DATASET must be vilp|hallusionbench|mmvet}
RUN_ID=${RUN_ID:?RUN_ID must identify this pipeline run}
cd "$PROJECT_ROOT"
mkdir -p logs/uq "results/uq/$RUN_ID/$MODEL"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "=== Running UQ: model=$MODEL dataset=$DATASET ==="
"$HF_PYTHON" scripts/uq/compute_uq.py \
  --greedy-input "results/generation/$RUN_ID/generation/$MODEL/greedy/$DATASET.jsonl" \
  --sample-input "results/generation/$RUN_ID/generation/$MODEL/samples/$DATASET.jsonl" \
  --output "results/uq/$RUN_ID/$MODEL/$DATASET.jsonl" \
  --entailment-model-path "$MODEL_ROOT/DeBERTa-v3-large-mnli-fever-anli-ling-wanli" \
  --entailment-batch-size 32 \
  --entailment-device cuda

echo "=== UQ completed successfully for $MODEL on $DATASET ==="
