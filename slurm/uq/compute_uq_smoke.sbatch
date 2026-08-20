#!/bin/bash
# Compute PPL, semantic entropy and UMPIRE for the 3x3x10 smoke matrix.
#SBATCH --job-name=uq-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/uq/%x-%j.out
#SBATCH --error=logs/uq/%x-%j.err

set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"
cd "$PROJECT_ROOT"
mkdir -p logs/uq
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
for model in llava qwen internvl; do
  for dataset in vilp hallusionbench mmvet; do
    "$HF_PYTHON" scripts/uq/compute_uq.py \
      --greedy-input "results/generation/$model/greedy/$dataset.jsonl" \
      --sample-input "results/generation/$model/samples/$dataset.jsonl" \
      --output "results/uq/$model/$dataset.jsonl" \
      --entailment-model-path "$MODEL_ROOT/DeBERTa-v3-large-mnli-fever-anli-ling-wanli" \
      --entailment-batch-size 32 --entailment-device cuda
  done
done
