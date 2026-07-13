#!/bin/bash
#SBATCH --job-name=vauq-exp2-analysis
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/lexiangrui/Uncertainty-Quantification-of-MLLM/logs/vauq/%x_%j.out
#SBATCH --error=/home/lexiangrui/Uncertainty-Quantification-of-MLLM/logs/vauq/%x_%j.err

set -euo pipefail
ROOT=/home/lexiangrui/Uncertainty-Quantification-of-MLLM
RESULTS=$ROOT/results/vauq/experiment2
cd "$ROOT/baseline/vauq-repro"

/home/lexiangrui/.venvs/vlm-transformers/bin/python -u scripts/analyze_experiment2.py \
  --mmvet "$RESULTS/mmvet_llava-1.5-7b_raw-question_core.mmhal-qwen36.jsonl" \
  --vilp "$RESULTS/vilp_llava-1.5-7b_raw-question_core.mmhal-qwen36.jsonl" \
  --output "$RESULTS/experiment2_analysis.json" \
  --bootstrap 2000
