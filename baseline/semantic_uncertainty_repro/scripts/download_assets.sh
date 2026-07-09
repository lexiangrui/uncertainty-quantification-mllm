#!/bin/bash
# Download assets for semantic uncertainty reproduction on mg01.
#
# Run this ON mg01 (which has internet access).  Compute nodes are offline.
#
# Usage:
#   ssh mg01
#   bash scripts/download_assets.sh
#
# This downloads:
#   1. Llama-2-7b-chat-hf  (~13 GB) — the LLM
#   2. DeBERTa-v2-xlarge-mnli (~1.7 GB) — the entailment model
#   3. TriviaQA dataset — auto-cached by the first load_dataset call
#
# Prerequisites:
#   - huggingface-cli login  (with a token that has LLaMA-2 access)
#   - HF_ENDPOINT=https://hf-mirror.com (recommended for speed in China)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# --- Load env vars if available ---
if [ -f "${REPO_ROOT}/configs/sem_unc.env" ]; then
    source "${REPO_ROOT}/configs/sem_unc.env"
fi

# --- Configurable paths ---
ASSETS_DIR="${SEM_UNC_ASSETS_DIR:-/opt/${USER}/sem_unc_assets}"
MODELS_DIR="${SEM_UNC_MODELS_DIR:-${ASSETS_DIR}/models}"
DATASETS_CACHE="${SEM_UNC_DATASETS_DIR:-${ASSETS_DIR}/datasets}"
HF_CACHE="${HF_HOME:-${ASSETS_DIR}/hf_cache}"

mkdir -p "${MODELS_DIR}" "${DATASETS_CACHE}" "${HF_CACHE}"

export HF_HOME="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}/hub"
export HF_DATASETS_CACHE="${DATASETS_CACHE}"

echo "========================================"
echo "Model dir:   ${MODELS_DIR}"
echo "Dataset dir: ${DATASETS_CACHE}"
echo "HF cache:    ${HF_CACHE}"
echo "========================================"

# ---- 1. Llama-2-7b-chat-hf ----
echo ""
echo ">>> Downloading Llama-2-7b-chat-hf ..."
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = 'meta-llama/Llama-2-7b-chat-hf'
save_dir = '${MODELS_DIR}/Llama-2-7b-chat-hf'

print(f'Downloading tokenizer to {save_dir}...')
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(save_dir)
print('Tokenizer saved.')

print(f'Downloading model to {save_dir}...')
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.save_pretrained(save_dir)
print('Model saved.')
"
echo "Llama-2-7b-chat-hf downloaded OK."

# ---- 2. DeBERTa entailment model ----
echo ""
echo ">>> Downloading DeBERTa-v2-xlarge-mnli ..."
python3 -c "
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = 'microsoft/deberta-v2-xlarge-mnli'
save_dir = '${MODELS_DIR}/deberta-v2-xlarge-mnli'

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(save_dir)
print('Tokenizer saved.')

model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.save_pretrained(save_dir)
print('Model saved.')
"
echo "DeBERTa downloaded OK."

# ---- 3. TriviaQA dataset (seed cache) ----
echo ""
echo ">>> Pre-caching TriviaQA dataset ..."
python3 -c "
from datasets import load_dataset
ds = load_dataset('TimoImhof/TriviaQA-in-SQuAD-format')
print(f'TriviaQA loaded: {ds}')
"
echo "TriviaQA cached OK."

echo ""
echo "========================================"
echo "All assets downloaded to ${ASSETS_DIR}"
echo "========================================"
echo ""
echo "Next step: submit the Slurm job:"
echo "  sbatch slurm/run_reproduce_triviaqa.sbatch"
