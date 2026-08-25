#!/bin/bash
# Interactive-companion variant of run_era.sbatch: run all three models
# sequentially inside one allocation (srun), no SBATCH directives.
# Usage: srun --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=08:00:00 \
#            bash slurm/improvement/run_era_all.sh
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT
source "$PROJECT_ROOT/slurm/common.sh"
cd "$PROJECT_ROOT"
mkdir -p logs/era

for MODEL in llava qwen internvl; do
  echo "==================== ERA model=$MODEL ===================="
  MODEL="$MODEL" bash "$PROJECT_ROOT/slurm/improvement/run_era.sbatch"
done
echo "ERA_ALL_DONE"
