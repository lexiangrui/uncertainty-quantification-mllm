#!/bin/bash
#SBATCH --job-name=vlu-mmvet-llava
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/vlu-mmvet-llava_%j.out
#SBATCH --error=logs/vlu-mmvet-llava_%j.err

# ===================================================================
# VL-Uncertainty: MM-Vet + LLaVA1.5-7B + Qwen2.5-3B-Instruct
# Reproduces Table 1 in Zhang et al. (2024)
# Expected hallucination detection accuracy: ~82.11%
# ===================================================================

set -euo pipefail

REPO_ROOT="/home/${USER}/vl_uncertainty_repro"
cd "${REPO_ROOT}"

# Load env config
if [ -f configs/vl_uncertainty.env ]; then
    set -a
    source configs/vl_uncertainty.env
    set +a
fi

# Offline mode — compute nodes have no outbound internet
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PYTHON="${PYTHON_BIN:-/home/${USER}/.venvs/vlm-transformers/bin/python}"

mkdir -p logs results

echo "============================================"
echo "VL-Uncertainty: MM-Vet + LLaVA1.5-7B"
echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "LLaVA model: ${VLU_MODEL_PATH}"
echo "Qwen model:  ${VLU_TEXT_MODEL_PATH}"
echo "============================================"

# Verify model paths
if [ ! -d "${VLU_MODEL_PATH}" ]; then
    echo "ERROR: LLaVA model not found at ${VLU_MODEL_PATH}"
    exit 1
fi
if [ ! -d "${VLU_TEXT_MODEL_PATH}" ]; then
    echo "ERROR: Qwen model not found at ${VLU_TEXT_MODEL_PATH}"
    exit 1
fi

# Run VL-Uncertainty on MM-Vet (218 samples)
# LLaVA → GPU 0, Qwen → GPU 1 (32GB each, avoid OOM)
$PYTHON scripts/run_vl_uncertainty.py \
    --backend llava \
    --model-path "${VLU_MODEL_PATH}" \
    --llava-device cuda:0 \
    --benchmark mmvet \
    --text-model qwen \
    --text-model-path "${VLU_TEXT_MODEL_PATH}" \
    --text-device cuda:1 \
    --judge llm \
    --sampling-temp 1.0 \
    --max-new-tokens 128 \
    --limit 218 \
    --seed 42 \
    --output results/mmvet_llava_vlu.jsonl

echo ""
echo "============================================"
echo "Job finished at $(date)"
echo "Results: ${REPO_ROOT}/results/mmvet_llava_vlu.jsonl"
echo "Summary: ${REPO_ROOT}/results/mmvet_llava_vlu.summary.json"
echo "============================================"
