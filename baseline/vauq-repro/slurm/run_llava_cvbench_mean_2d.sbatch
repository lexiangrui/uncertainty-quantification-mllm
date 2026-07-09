#!/bin/bash
#SBATCH --job-name=vauq-mean-2d
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=25
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /home/lexiangrui/Uncertainty-Quantification-of-MLLM/baseline/vauq-repro
mkdir -p logs results

export VAUQ_ASSETS_DIR="/opt/lexiangrui/vauq_assets"
export VAUQ_MODELS_DIR="${VAUQ_ASSETS_DIR}/models"
export VAUQ_DATASETS_DIR="${VAUQ_ASSETS_DIR}/datasets"
export LLAVA_HF_MODEL="${VAUQ_MODELS_DIR}/llava-1.5-7b-hf"
export PYTHON_BIN="/home/lexiangrui/.venvs/vlm-transformers/bin/python3"

export HF_HOME="/opt/lexiangrui/hf_cache"
export HF_HUB_CACHE="/opt/lexiangrui/hf_cache/hub"
export TRANSFORMERS_CACHE="/opt/lexiangrui/hf_cache/hub"
export HF_MODULES_CACHE="/opt/lexiangrui/hf_cache/modules"
export HF_DATASETS_CACHE="${VAUQ_DATASETS_DIR}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LIMIT="${LIMIT:-1438}"
OUT="${OUTPUT:-results/cvbench_llava-1.5-7b-hf_attention_mean_limit${LIMIT}.jsonl}"

echo "============================================"
echo "VAUQ attention core + mean ablation"
echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo "Model: ${LLAVA_HF_MODEL}"
echo "Limit: ${LIMIT}"
echo "Output: ${OUT}"
echo "============================================"

"${PYTHON_BIN}" scripts/run_vauq.py \
    --backend llava \
    --benchmark cvbench \
    --judge letter \
    --model-path "${LLAVA_HF_MODEL}" \
    --mask-strategy core \
    --ablation-baseline mean \
    --inference-temp 0.0 \
    --max-new-tokens 128 \
    --limit "${LIMIT}" \
    --output "${OUT}"

echo "============================================"
echo "Job finished at $(date)"
echo "Results: ${OUT}"
echo "Summary: ${OUT%.jsonl}.summary.json"
echo "============================================"
