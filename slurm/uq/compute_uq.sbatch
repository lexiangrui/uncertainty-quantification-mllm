#!/bin/bash
#SBATCH --job-name=uq
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/uq/%x-%j.out
#SBATCH --error=logs/uq/%x-%j.out

# Standard UQ computation (PPL / SE / UMPIRE) over generation outputs.
# Usage: sbatch --export=MODEL=llava,RUN_ID=my-run slurm/uq/compute_uq.sbatch
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

MODEL=${MODEL:?MODEL must be llava|qwen|internvl}
RUN_ID=${RUN_ID:?RUN_ID must identify this pipeline run}
cd "$PROJECT_ROOT"
mkdir -p logs/uq "results/uq/$RUN_ID/$MODEL"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

for dataset in vilp hallusionbench mmvet; do
  echo "=== Running UQ: model=$MODEL dataset=$dataset ==="
  "$HF_PYTHON" scripts/uq/compute_uq.py \
    --greedy-input "results/generation/$RUN_ID/generation/$MODEL/greedy/$dataset.jsonl" \
    --sample-input "results/generation/$RUN_ID/generation/$MODEL/samples/$dataset.jsonl" \
    --output "results/uq/$RUN_ID/$MODEL/$dataset.jsonl" \
    --entailment-model-path "$MODEL_ROOT/DeBERTa-v3-large-mnli-fever-anli-ling-wanli" \
    --entailment-batch-size 32 \
    --entailment-device cuda
done
