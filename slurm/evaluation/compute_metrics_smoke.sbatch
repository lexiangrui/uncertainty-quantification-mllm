#!/bin/bash
# Compute all UQ-vs-label metrics for the 3x3x10 smoke matrix.
#SBATCH --job-name=metrics-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/evaluation/%x-%j.out
#SBATCH --error=logs/evaluation/%x-%j.err

set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"
cd "$PROJECT_ROOT"
mkdir -p logs/evaluation
for model in llava qwen internvl; do
  for dataset in vilp hallusionbench mmvet; do
    "$HF_PYTHON" scripts/evaluation/compute_metrics.py \
      --uq-input "results/uq/$model/$dataset.jsonl" \
      --judge-input "results/judging/$model/$dataset.jsonl" \
      --output "results/metrics/$model/$dataset.json" \
      --bootstrap-samples 100
  done
done
