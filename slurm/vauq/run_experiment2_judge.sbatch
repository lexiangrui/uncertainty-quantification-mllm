#!/bin/bash
#SBATCH --job-name=vauq-exp2-judge
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1%1
#SBATCH --output=/home/lexiangrui/Uncertainty-Quantification-of-MLLM/logs/vauq/%x_%A_%a.out
#SBATCH --error=/home/lexiangrui/Uncertainty-Quantification-of-MLLM/logs/vauq/%x_%A_%a.err

set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/opt/lexiangrui/hf_cache
export HF_DATASETS_CACHE=/opt/lexiangrui/datasets
export VAUQ_DATASETS_DIR=/opt/lexiangrui/datasets
export VILP_PARQUET_PATH=/opt/lexiangrui/datasets/vilp/ViLP.parquet
export QWEN_MMHAL_MODEL=/opt/lexiangrui/models/Qwen3.6-27B
export PYTHONUNBUFFERED=1

ROOT=/home/lexiangrui/Uncertainty-Quantification-of-MLLM
DATASETS=(mmvet vilp)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
INPUT=$ROOT/results/vauq/experiment2/${DATASET}_llava-1.5-7b_raw-question_core.jsonl
OUTPUT=$ROOT/results/vauq/experiment2/${DATASET}_llava-1.5-7b_raw-question_core.mmhal-qwen36.jsonl
cd "$ROOT/baseline/vauq-repro"

/home/lexiangrui/.venvs/vlm-transformers/bin/python -u scripts/judge_hallucination_mmhal.py \
  --benchmark "$DATASET" --input "$INPUT" --output "$OUTPUT" \
  --model-path "$QWEN_MMHAL_MODEL" --resume
