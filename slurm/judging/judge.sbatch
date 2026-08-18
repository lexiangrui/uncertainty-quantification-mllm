#!/bin/bash
#SBATCH --job-name=judge
#SBATCH --partition=batch,debug01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/judging/%x-%j.out
#SBATCH --error=logs/judging/%x-%j.out

# Standard LLM Judge execution over greedy responses (with default 10 concurrency).
# Usage: sbatch --export=MODEL=llava slurm/judging/judge.sbatch
set -euo pipefail

MODEL=${MODEL:?MODEL must be llava|qwen|internvl}
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-2024-08-06}"
cd /home/lexiangrui/Uncertainty-Quantification-of-MLLM
mkdir -p logs/judging "results/judging/$MODEL"

for dataset in vilp hallusionbench mmvet; do
  case $dataset in
    vilp) SRC=/opt/lexiangrui/datasets/vilp/ViLP.parquet ;;
    hallusionbench) SRC=/opt/lexiangrui/datasets/HallusionBench/data ;;
    mmvet) SRC=/opt/lexiangrui/datasets/MMVet/data/test-00000-of-00001.parquet ;;
  esac

  echo "=== Running Judge: model=$MODEL dataset=$dataset judge=$JUDGE_MODEL ==="
  ~/.venvs/vlm-transformers/bin/python scripts/judging/judge_responses.py \
    --dataset "$dataset" \
    --dataset-source "$SRC" \
    --greedy-input "results/generation/$MODEL/greedy/$dataset.jsonl" \
    --output "results/judging/$MODEL/$dataset.jsonl" \
    --model "$JUDGE_MODEL" \
    --concurrency 10
done
