#!/bin/bash
#SBATCH --job-name=vl_sem_unc_verify
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/vl_sem_unc_verify_%j.out
#SBATCH --error=logs/vl_sem_unc_verify_%j.err

# ===================================================================
# Verify: TriviaQA + Mistral-7B-Instruct-v0.1 + official semantic entropy
# ===================================================================

set -euo pipefail

REPO_ROOT="/home/${USER}/vl_uncertainty_repro"
cd "${REPO_ROOT}"

# Load env config
if [ -f configs/vl_uncertainty.env ]; then
    source configs/vl_uncertainty.env
fi

# Offline mode
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="/opt/lexiangrui/hf_cache"
export HF_HUB_CACHE="/opt/lexiangrui/hf_cache/hub"

DEBERTA_PATH="/opt/lexiangrui/sem_unc_assets/models/deberta-v2-xlarge-mnli"
MODEL_PATH="/opt/lexiangrui/sem_unc_assets/models/Mistral-7B-Instruct-v0.1"

if [ -d "${DEBERTA_PATH}" ]; then
    export DEBERTA_MNLI_MODEL="${DEBERTA_PATH}"
fi

PYTHON=/home/lexiangrui/.venvs/vlm-transformers/bin/python

mkdir -p logs results

echo "============================================"
echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "Model: ${MODEL_PATH}"
echo "DeBERTa: ${DEBERTA_MNLI_MODEL:-hub}"
echo "============================================"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model not found at ${MODEL_PATH}"
    exit 1
fi

$PYTHON scripts/run_vl_uncertainty.py \
    --backend huggingface_llm \
    --model-path "${MODEL_PATH}" \
    --benchmark trivia_qa \
    --num-generations 10 \
    --sampling-temp 1.0 \
    --max-new-tokens 50 \
    --strict-entailment \
    --limit 400 \
    --seed 10 \
    --output results/verify_mistral_triviaqa.jsonl

echo ""
echo "============================================"
echo "Job finished at $(date)"
echo "Results: results/verify_mistral_triviaqa.jsonl"
echo "Summary: results/verify_mistral_triviaqa.summary.json"
echo "============================================"
