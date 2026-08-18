#!/bin/bash
# Submit 9-GPU Full Grid Parallel Generation + Chained UQ Computation
# Models: llava_1_5, qwen2_5_vl, internvl3_5
# Datasets: vilp, hallusionbench, mmvet
# Total: 9 Generation Jobs (Concurrent) -> 9 UQ Jobs (Chained via afterok)
set -euo pipefail

cd /home/lexiangrui/Uncertainty-Quantification-of-MLLM
mkdir -p logs/{generation,uq}

echo "================================================================="
echo "   Submitting 9-GPU Concurrent Generation & Chained UQ Pipeline   "
echo "================================================================="

MODELS=(llava qwen internvl)
DATASETS=(vilp hallusionbench mmvet)

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    # 1. Submit Generation Job (1 GPU)
    GEN_OUT=$(sbatch --export=MODEL="$MODEL",DATASET="$DATASET" slurm/generation/generate_single.sbatch)
    GEN_ID=$(echo "$GEN_OUT" | awk '{print $4}')

    # 2. Submit Chained UQ Computation Job (1 GPU, triggers immediately after Generation finishes)
    UQ_OUT=$(sbatch --dependency=afterok:"$GEN_ID" --export=MODEL="$MODEL",DATASET="$DATASET" slurm/uq/compute_uq_single.sbatch)
    UQ_ID=$(echo "$UQ_OUT" | awk '{print $4}')

    printf "  • [%-8s x %-14s] Gen Job ID: %-8s -> UQ Job ID: %-8s (afterok:%s)\n" "$MODEL" "$DATASET" "$GEN_ID" "$UQ_ID" "$GEN_ID"
  done
done

echo "================================================================="
echo "All 9 Generation + 9 UQ jobs scheduled! Monitor with: squeue -u \$USER"
echo "================================================================="
