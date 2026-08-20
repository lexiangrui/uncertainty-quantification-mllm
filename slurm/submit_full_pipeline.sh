#!/bin/bash
# Submit full automated 4-stage pipeline for MLLM Uncertainty Quantification
# Stage 1: Generation (Greedy + K=10 Samples with .tokens and hidden sidecars)
# Stage 2: Baseline UQ (PPL / SE / UMPIRE via DeBERTa) [depends on Stage 1]
# The production DAG per model is: Hugging Face generation -> UQ.
# Judge and ERA are submitted separately because they consume API quota and an
# additional GPU, respectively.
#
# Usage:
#   bash slurm/submit_full_pipeline.sh          # Submit for all 3 models (llava, qwen, internvl)
#   bash slurm/submit_full_pipeline.sh llava    # Submit for a single model
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if (($#)); then
  MODELS=("$@")
else
  MODELS=(llava qwen internvl)
fi
cd "$PROJECT_ROOT"
mkdir -p logs/{generation,uq,judging,era}

echo "================================================================="
echo "   Submitting Full MLLM Uncertainty Quantification Pipeline      "
echo "   Models: ${MODELS[*]}"
echo "================================================================="

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo ">>> [Model: $MODEL] Scheduling Hugging Face generation -> UQ..."

  # Generation writes response, token, log-probability, and hidden-state artifacts.
  GEN_ID=$(sbatch --parsable --export=ALL,MODEL="$MODEL" slurm/generation/generate.sbatch)
  echo "  [Generation]             $GEN_ID"

  # UQ starts only after all generation artifacts are complete.
  UQ_ID=$(sbatch --parsable --dependency=afterok:"$GEN_ID" --export=ALL,MODEL="$MODEL" slurm/uq/compute_uq.sbatch)
  echo "  [UQ]                     $UQ_ID (afterok:$GEN_ID)"
done

echo ""
echo "================================================================="
echo "All jobs successfully scheduled! Check status with: squeue -u \$USER"
echo "================================================================="
