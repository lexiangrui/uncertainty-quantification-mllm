#!/bin/bash
#SBATCH --job-name=vauq-qwen-judge
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-3%1
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
fi

export PATH="/opt/slurm/bin:$PATH"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results

source configs/vauq.env

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

TASKS=(
    "results/vilp_llava-1.5-7b-hf_paper_v2.jsonl"
    "results/mmvet_llava-1.5-7b-hf_paper_v2.jsonl"
    "results/vilp_llava-1.5-13b-hf_paper_v2.jsonl"
    "results/mmvet_llava-1.5-13b-hf_paper_v2.jsonl"
)

IN_PATH="${TASKS[$SLURM_ARRAY_TASK_ID]}"
OUT_PATH="${IN_PATH%.jsonl}.qwen_judged.jsonl"

echo "qwen_judge_model=${QWEN_JUDGE_MODEL}"
echo "input=${IN_PATH}"
echo "output=${OUT_PATH}"

"$PYTHON_BIN" scripts/apply_llm_judge.py \
    --judge qwen_local \
    --input "$IN_PATH" \
    --output "$OUT_PATH" \
    --resume
