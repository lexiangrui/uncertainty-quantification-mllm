#!/bin/bash
#SBATCH --job-name=generate-responses
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/generation/%x_%j.out
#SBATCH --error=logs/generation/%x_%j.err

set -euo pipefail

# The resource header is the conservative one-GPU profile for the three 7B/8B models.

: "${PROJECT_ROOT:?set PROJECT_ROOT to the repository path}"
: "${PYTHON_BIN:?set PYTHON_BIN to the project Python interpreter}"
: "${DATASET:?set DATASET to vilp, hallusionbench, or mmvet}"
: "${DATASET_SOURCE:?set DATASET_SOURCE to the downloaded server asset}"
: "${MODEL_FAMILY:?set MODEL_FAMILY to llava_1_5, qwen2_5_vl, or internvl3_5}"
: "${MODEL_PATH:?set MODEL_PATH to the downloaded model directory}"
: "${ADAPTER_PATH:?set ADAPTER_PATH to the trained LoRA adapter directory}"
: "${OUTPUT:?set OUTPUT to the server JSONL output path}"
: "${GENERATION_PHASE:?set GENERATION_PHASE to greedy or samples}"
SRUN_BIN="${SRUN_BIN:-/opt/slurm/bin/srun}"
if [[ ! -x "$SRUN_BIN" ]]; then
    echo "Slurm launcher is not executable: $SRUN_BIN" >&2
    exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

cd "$PROJECT_ROOT"
mkdir -p logs/generation "$(dirname "$OUTPUT")"

extra_args=()
if [[ -n "${LIMIT:-}" ]]; then
    extra_args+=(--limit "$LIMIT")
fi
if [[ -n "${SAMPLE_IDS_FILE:-}" ]]; then
    extra_args+=(--sample-ids-file "$SAMPLE_IDS_FILE")
fi
"$SRUN_BIN" "$PYTHON_BIN" scripts/generation/generate_responses.py \
    --dataset "$DATASET" \
    --dataset-source "$DATASET_SOURCE" \
    --model-family "$MODEL_FAMILY" \
    --model-path "$MODEL_PATH" \
    --adapter-path "$ADAPTER_PATH" \
    --output "$OUTPUT" \
    --phase "$GENERATION_PHASE" \
    --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
    --num-samples "${NUM_SAMPLES:-10}" \
    --seed "${SEED:-42}" \
    --reject-resample-k "${REJECT_RESAMPLE_K:-50}" \
    --max-batch-size "${MAX_BATCH_SIZE:-5}" \
    --request-window-samples "${REQUEST_WINDOW_SAMPLES:-16}" \
    --attn-implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
    "${extra_args[@]}"
