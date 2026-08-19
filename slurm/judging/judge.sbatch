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
# Usage: sbatch --export=MODEL=llava,RUN_ID=my-run slurm/judging/judge.sbatch
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"

MODEL=${MODEL:?MODEL must be llava|qwen|internvl}
RUN_ID=${RUN_ID:?RUN_ID must identify this pipeline run}
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-2024-08-06}"
cd "$PROJECT_ROOT"
mkdir -p logs/judging "results/judging/$RUN_ID/$MODEL"

for dataset in vilp hallusionbench mmvet; do
  case $dataset in
    vilp) SRC="$DATA_ROOT/vilp/ViLP.parquet" ;;
    hallusionbench) SRC="$DATA_ROOT/HallusionBench/data" ;;
    mmvet) SRC="$DATA_ROOT/MMVet/data/test-00000-of-00001.parquet" ;;
  esac

  echo "=== Running Judge: model=$MODEL dataset=$dataset judge=$JUDGE_MODEL ==="
  "$HF_PYTHON" scripts/judging/judge_responses.py \
    --dataset "$dataset" \
    --dataset-source "$SRC" \
    --greedy-input "results/generation/$RUN_ID/generation/$MODEL/greedy/$dataset.jsonl" \
    --output "results/judging/$RUN_ID/$MODEL/$dataset.jsonl" \
    --model "$JUDGE_MODEL" \
    --concurrency 10
done
