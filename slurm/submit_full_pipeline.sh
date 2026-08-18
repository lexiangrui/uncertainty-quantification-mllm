#!/bin/bash
# Submit full automated 4-stage pipeline for MLLM Uncertainty Quantification
# Stage 1: Generation (Greedy + K=10 Samples with .tokens and hidden sidecars)
# Stage 2: Baseline UQ (PPL / SE / UMPIRE via DeBERTa) [depends on Stage 1]
# Stage 3: LLM Judge (GPT-4o 10-worker concurrency) [depends on Stage 1]
# Stage 4: ERA Attention Feature Extraction (Layer 0-1) [depends on Stage 1]
#
# Usage:
#   bash slurm/submit_full_pipeline.sh          # Submit for all 3 models (llava, qwen, internvl)
#   bash slurm/submit_full_pipeline.sh llava    # Submit for a single model
set -euo pipefail

MODELS=("${@:-llava qwen internvl}")

cd /home/lexiangrui/Uncertainty-Quantification-of-MLLM
mkdir -p logs/{generation,uq,judging,era}

echo "================================================================="
echo "   Submitting Full MLLM Uncertainty Quantification Pipeline      "
echo "   Models: ${MODELS[*]}"
echo "================================================================="

for MODEL in ${MODELS[*]}; do
  echo ""
  echo ">>> [Model: $MODEL] Scheduling 2-GPU Generation + Chained UQ..."

  # Stage 1: Generation (2 GPUs per model, sequential datasets: vilp -> hallusionbench -> mmvet)
  GEN_OUT=$(sbatch --export=MODEL="$MODEL" slurm/generation/generate.sbatch)
  GEN_ID=$(echo "$GEN_OUT" | awk '{print $4}')
  echo "  [Stage 1: 2-GPU Generation] Submitted Job ID: $GEN_ID"

  # Stage 2: UQ Baseline (GPU, depends on Stage 1)
  UQ_OUT=$(sbatch --dependency=afterok:"$GEN_ID" --export=MODEL="$MODEL" slurm/uq/compute_uq.sbatch)
  UQ_ID=$(echo "$UQ_OUT" | awk '{print $4}')
  echo "  [Stage 2: Chained UQ]      Submitted Job ID: $UQ_ID (afterok:$GEN_ID)"
done

echo ""
echo "================================================================="
echo "All jobs successfully scheduled! Check status with: squeue -u \$USER"
echo "================================================================="
