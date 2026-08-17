#!/bin/bash
#SBATCH --job-name=compute-uq
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# NLI entailment runs through Transformers and should use CUDA when requested.
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/generation/%x_%j.out
#SBATCH --error=logs/generation/%x_%j.err

set -euo pipefail

: "${PROJECT_ROOT:?set PROJECT_ROOT}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${GREEDY_INPUT:?set GREEDY_INPUT}"
: "${SAMPLE_INPUT:?set SAMPLE_INPUT}"
: "${OUTPUT:?set OUTPUT}"
: "${ENTAILMENT_MODEL_PATH:?set ENTAILMENT_MODEL_PATH}"

cd "$PROJECT_ROOT"
mkdir -p logs/generation "$(dirname "$OUTPUT")"
"${SRUN_BIN:-/opt/slurm/bin/srun}" "$PYTHON_BIN" scripts/uq/compute_uq.py \
    --greedy-input "$GREEDY_INPUT" \
    --sample-input "$SAMPLE_INPUT" \
    --output "$OUTPUT" \
    --entailment-model-path "$ENTAILMENT_MODEL_PATH" \
    --entailment-batch-size "${ENTAILMENT_BATCH_SIZE:-32}" \
    --entailment-device "${ENTAILMENT_DEVICE:-cpu}"
