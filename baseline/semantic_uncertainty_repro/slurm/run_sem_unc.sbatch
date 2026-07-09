#!/bin/bash
#SBATCH --job-name=sem_unc
#SBATCH --output=logs/sem_unc_%A_%a.out
#SBATCH --error=logs/sem_unc_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-5

# Source environment configuration.
source configs/sem_unc.env

# Enable offline mode for compute nodes.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Paper grid: 3 datasets × 2 models = 6 combinations.
# Modify MODELS / DATASETS arrays as needed.
MODELS=("llava-1.5-7b-hf" "llava-1.5-13b-hf")
DATASETS=("cvbench" "mmvet" "vilp")

N_MODELS=${#MODELS[@]}
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / ${#DATASETS[@]}))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID % ${#DATASETS[@]}))

MODEL="${MODELS[$MODEL_IDX]}"
DATASET="${DATASETS[$DATASET_IDX]}"
MODEL_PATH="${SEM_UNC_MODELS_DIR}/${MODEL}"

echo "==> Model: ${MODEL}  Dataset: ${DATASET}"
echo "==> Model path: ${MODEL_PATH}"

python scripts/run_semantic_uncertainty.py \
    --model llava \
    --model-path "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --num-generations 10 \
    --num-samples 400 \
    --output-dir "results/sem_unc" \
    --tag "${MODEL}_${DATASET}"
