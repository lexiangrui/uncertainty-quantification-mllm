#!/bin/bash
# Deterministic, offline labels for the 10-sample hybrid smoke outputs.
#SBATCH --job-name=judge-ref-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/judging/%x-%j.out
#SBATCH --error=logs/judging/%x-%j.err

set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"
cd "$PROJECT_ROOT"
mkdir -p logs/judging
for model in llava qwen internvl; do
  for dataset in vilp hallusionbench mmvet; do
    case "$dataset" in
      vilp) SRC="$DATA_ROOT/vilp/ViLP.parquet" ;;
      hallusionbench) SRC="$DATA_ROOT/HallusionBench/data" ;;
      mmvet) SRC="$DATA_ROOT/MMVet/data/test-00000-of-00001.parquet" ;;
    esac
    "$HF_PYTHON" scripts/judging/judge_reference_rule.py \
      --dataset "$dataset" --dataset-source "$SRC" \
      --greedy-input "results/generation/$model/greedy/$dataset.jsonl" \
      --output "results/judge/$model/$dataset.jsonl" --limit 10
  done
done
