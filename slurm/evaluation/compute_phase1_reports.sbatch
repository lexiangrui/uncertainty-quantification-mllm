#!/bin/bash
#SBATCH --job-name=phase1-reports
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/evaluation/%x_%j.out
#SBATCH --error=logs/evaluation/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/lexiangrui/Uncertainty-Quantification-of-MLLM}"
PYTHON_BIN="${PYTHON_BIN:-/home/lexiangrui/.venvs/vlm-transformers/bin/python}"

cd "$PROJECT_ROOT"
mkdir -p logs/evaluation results/metrics results/analysis

for model in llava qwen internvl; do
    for dataset in vilp hallusionbench mmvet; do
        "$PYTHON_BIN" scripts/evaluation/compute_metrics.py \
            --uq-input "results/uq/$model/$dataset.jsonl" \
            --judge-input "results/judging/$model/$dataset.jsonl" \
            --output "results/metrics/$model/$dataset.json"
    done
done

"$PYTHON_BIN" scripts/analysis/analyze_phase1_results.py \
    --project-root "$PROJECT_ROOT" \
    --output-dir results/analysis

"$PYTHON_BIN" scripts/analysis/compute_common_luh.py
