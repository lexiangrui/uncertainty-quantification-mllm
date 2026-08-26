#!/bin/bash
#SBATCH --job-name=xml-sample
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ablation/%x-%j.out
#SBATCH --error=logs/ablation/%x-%j.out

set -euo pipefail
trap 'echo "SCRIPT FAILED at line $LINENO (exit $?)" >&2' ERR
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"

cd "$PROJECT_ROOT"
mkdir -p logs/ablation
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

"$HF_PYTHON" scripts/ablation/prepare_xml_format_ablation.py \
  --data-root "$DATA_ROOT" \
  --sample-size 500 \
  --seed 42
