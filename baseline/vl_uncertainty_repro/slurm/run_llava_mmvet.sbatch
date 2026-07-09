#!/bin/bash
#SBATCH --job-name=vlu-llava-mmvet
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/vlu-llava-mmvet_%j.out
#SBATCH --error=logs/vlu-llava-mmvet_%j.err

set -euo pipefail

cd /home/lexiangrui/vl_uncertainty_repro
mkdir -p logs results
source configs/vl_uncertainty.env

"${PYTHON_BIN:-python}" scripts/run_vl_uncertainty.py \
  --backend llava \
  --benchmark mmvet \
  --text-model qwen \
  --judge llm \
  --model-path "$VLU_MODEL_PATH" \
  --text-model-path "$VLU_TEXT_MODEL_PATH" \
  --limit "${LIMIT:-0}" \
  --output results/mmvet_llava_vlu.jsonl
