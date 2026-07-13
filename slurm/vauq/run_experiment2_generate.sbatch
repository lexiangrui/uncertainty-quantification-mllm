#!/bin/bash
#SBATCH --job-name=vauq-exp2-gen
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/lexiangrui/Uncertainty-Quantification-of-MLLM/logs/vauq/%x_%A_%a.out
#SBATCH --error=/home/lexiangrui/Uncertainty-Quantification-of-MLLM/logs/vauq/%x_%A_%a.err

set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/opt/lexiangrui/hf_cache
export HF_DATASETS_CACHE=/opt/lexiangrui/datasets
export VAUQ_DATASETS_DIR=/opt/lexiangrui/datasets
export VILP_PARQUET_PATH=/opt/lexiangrui/datasets/vilp/ViLP.parquet
export PYTHONUNBUFFERED=1

ROOT=/home/lexiangrui/Uncertainty-Quantification-of-MLLM
DATASETS=(mmvet vilp)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
OUTPUT=$ROOT/results/vauq/experiment2/${DATASET}_llava-1.5-7b_raw-question_core.jsonl
mkdir -p "$ROOT/results/vauq/experiment2" "$ROOT/logs/vauq"
cd "$ROOT/baseline/vauq-repro"

/home/lexiangrui/.venvs/vlm-transformers/bin/python -u scripts/run_vauq.py \
  --backend llava --benchmark "$DATASET" --defer-judge \
  --model-path /opt/lexiangrui/models/llava-1.5-7b-hf \
  --question-style raw --mask-strategy core --max-new-tokens 128 \
  --resume --output "$OUTPUT"
