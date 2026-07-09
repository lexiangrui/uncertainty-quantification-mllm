#!/bin/bash
#SBATCH --job-name=sem_unc_triviaqa
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sem_unc_triviaqa_%j.out
#SBATCH --error=logs/sem_unc_triviaqa_%j.err

# ===================================================================
# Reproduce: TriviaQA + Llama-2-7b-chat + semantic uncertainty
#
# Paper reports for this combination (Farquhar et al., Nature 2024):
#   Semantic Entropy AUROC: ~0.79 (averaged across all 30 model×dataset
#   combinations; individual values range 0.78–0.81)
# ===================================================================

set -euo pipefail

REPO_ROOT="/home/${USER}/semantic_uncertainty_repro"
cd "${REPO_ROOT}"

# Load environment config.
if [ -f configs/sem_unc.env ]; then
    source configs/sem_unc.env
fi

# Offline mode — compute nodes have no internet.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT=https://hf-mirror.com

# Set local model paths (overrides HuggingFace hub lookups).
export DEBERTA_MNLI_MODEL="${SEM_UNC_MODELS_DIR:-/opt/${USER}/sem_unc_assets/models}/deberta-v2-xlarge-mnli"

# Ensure Python and dependencies are available.
module purge

mkdir -p logs results/sem_unc

echo "============================================"
echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not available)"
echo "============================================"

# ---- Run reproduction ----
python scripts/run_semantic_uncertainty.py \
    --model huggingface_llm \
    --model-path "${SEM_UNC_MODELS_DIR:-/opt/${USER}/sem_unc_assets/models}/Llama-2-7b-chat-hf" \
    --dataset trivia_qa \
    --num-generations 10 \
    --num-samples 400 \
    --temperature 1.0 \
    --max-new-tokens 50 \
    --metric squad \
    --entailment-model deberta \
    --strict-entailment \
    --seed 10 \
    --output-dir results/sem_unc \
    --tag "Llama-2-7b-chat_trivia_qa"

echo ""
echo "============================================"
echo "Job finished at $(date)"
echo "Results: results/sem_unc/Llama-2-7b-chat_trivia_qa.jsonl"
echo "Summary: results/sem_unc/Llama-2-7b-chat_trivia_qa.summary.json"
echo "============================================"
