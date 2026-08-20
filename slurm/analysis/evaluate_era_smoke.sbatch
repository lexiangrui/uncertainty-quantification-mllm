#!/bin/bash
# Summarize ERA AUROC/AUPRC/PRR for the 3x3x10 smoke matrix.
#SBATCH --job-name=era-metrics-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/era/%x-%j.out
#SBATCH --error=logs/era/%x-%j.err

set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"
cd "$PROJECT_ROOT"
mkdir -p logs/era
"$HF_PYTHON" scripts/analysis/evaluate_era_smoke.py \
  --results-root results --output results/metrics/era_smoke.json --layers 0 1
