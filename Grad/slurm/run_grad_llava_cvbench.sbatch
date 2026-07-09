#!/bin/bash
#SBATCH --job-name=grad-vauq-cvbench
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /home/lexiangrui/Uncertainty-Quantification-of-MLLM
mkdir -p logs results

export VAUQ_ASSETS_DIR="/opt/lexiangrui/vauq_assets"
export VAUQ_MODELS_DIR="${VAUQ_ASSETS_DIR}/models"
export VAUQ_DATASETS_DIR="${VAUQ_ASSETS_DIR}/datasets"
export LLAVA_7B_HF_MODEL="${VAUQ_MODELS_DIR}/llava-1.5-7b-hf"
export PYTHON_BIN="/home/lexiangrui/.venvs/vlm-transformers/bin/python3"

export HF_HOME="/opt/lexiangrui/hf_cache"
export HF_HUB_CACHE="/opt/lexiangrui/hf_cache/hub"
export TRANSFORMERS_CACHE="/opt/lexiangrui/hf_cache/hub"
export HF_MODULES_CACHE="/opt/lexiangrui/hf_cache/modules"
export HF_DATASETS_CACHE="${VAUQ_DATASETS_DIR}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

LIMIT="${LIMIT:-4}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
BASELINE="${BASELINE:-zero}"
SELECTOR="${SELECTOR:-grad_x_act}"

LIMIT_SUFFIX="full"
LIMIT_ARGS=()
if [[ "${LIMIT}" != "0" ]]; then
    LIMIT_SUFFIX="limit${LIMIT}"
    LIMIT_ARGS=(--limit "${LIMIT}")
fi

OUT="results/grad_cvbench_llava-1.5-7b-hf_${ATTN_IMPL}_${SELECTOR}_${BASELINE}_${LIMIT_SUFFIX}.jsonl"

echo "============================================"
echo "Grad-VAUQ: CVBench + LLaVA1.5-7B"
echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo "Model: ${LLAVA_7B_HF_MODEL}"
echo "Attention implementation: ${ATTN_IMPL}"
echo "Selector: ${SELECTOR}"
echo "Ablation baseline: ${BASELINE}"
echo "Limit: ${LIMIT}"
echo "Output: ${OUT}"
echo "============================================"

"${PYTHON_BIN}" Grad/scripts/run_grad_vauq.py \
    --backend llava \
    --visual-adapter llava \
    --benchmark cvbench \
    --judge letter \
    --model-path "${LLAVA_7B_HF_MODEL}" \
    --attn-implementation "${ATTN_IMPL}" \
    --selector "${SELECTOR}" \
    --ablation-baseline "${BASELINE}" \
    --max-new-tokens 128 \
    --output "${OUT}" \
    "${LIMIT_ARGS[@]}"

echo "============================================"
echo "Job finished at $(date)"
echo "Results: ${OUT}"
echo "Summary: ${OUT%.jsonl}.summary.json"
echo "============================================"
