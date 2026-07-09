#!/bin/bash
#SBATCH --job-name=sem_unc_mistral_official
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sem_unc_mistral_official_%j.out
#SBATCH --error=logs/sem_unc_mistral_official_%j.err

set -euo pipefail

REPO_ROOT="/home/${USER}/semantic_uncertainty_repro"
cd "${REPO_ROOT}"

source configs/sem_unc.env

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/opt/${USER}/hf_cache"
export HF_HUB_CACHE="/opt/${USER}/hf_cache/hub"
export HF_DATASETS_CACHE="/opt/${USER}/hf_cache/datasets"
export DEBERTA_MNLI_MODEL="${SEM_UNC_MODELS_DIR}/deberta-v2-xlarge-mnli"

PYTHON="${PYTHON_BIN:-/home/${USER}/.venvs/vlm-transformers/bin/python}"
MODEL_PATH="${SEM_UNC_MODELS_DIR}/Mistral-7B-Instruct-v0.1"
TAG="Mistral-7B-Instruct_trivia_qa_official"

mkdir -p logs results/sem_unc

echo "============================================"
echo "Job started at $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi N/A"
echo "Python: ${PYTHON}"
echo "Model: ${MODEL_PATH}"
echo "Tag: ${TAG}"
echo "============================================"

"${PYTHON}" scripts/run_semantic_uncertainty.py \
    --model huggingface_llm \
    --model-path "${MODEL_PATH}" \
    --dataset trivia_qa \
    --num-generations 10 \
    --num-samples 400 \
    --num-few-shot 5 \
    --temperature 1.0 \
    --max-new-tokens 50 \
    --metric squad \
    --entailment-model deberta \
    --strict-entailment \
    --seed 10 \
    --output-dir results/sem_unc \
    --tag "${TAG}"

echo ""
echo "============================================"
echo "Job finished at $(date)"
echo "Results: results/sem_unc/${TAG}.jsonl"
echo "Summary: results/sem_unc/${TAG}.summary.json"
echo "============================================"
