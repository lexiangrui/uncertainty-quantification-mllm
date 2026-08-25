#!/bin/bash
#SBATCH --job-name=downstream
#SBATCH --partition=batch,debug01
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/downstream-sbatch-%j.out
#SBATCH --error=logs/downstream-sbatch-%j.out

# Full downstream rerun on the 2026-08-22 regenerated samples:
# UQ -> metrics -> LUH subsets -> ERA -> ERA evaluation, all in this one allocation.
set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"
cd "$PROJECT_ROOT"
mkdir -p logs
bash scripts/run_downstream_20260822.sh
