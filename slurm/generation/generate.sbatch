#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --partition=batch,debug01
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/generation/%x-%j.out
#SBATCH --error=logs/generation/%x-%j.out

# Full Hugging Face generation: datasets run sequentially and each writes
# greedy token sidecars plus sampled token/hidden-state sidecars inline.
#
# Usage: sbatch --export=MODEL=llava slurm/generation/generate.sbatch
set -euo pipefail
trap 'echo "SCRIPT FAILED at line $LINENO (exit $?)" >&2' ERR
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"

MODEL=${MODEL:?MODEL must be llava|qwen|internvl}
cd "$PROJECT_ROOT"
mkdir -p logs/generation

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON_BIN="$HF_PYTHON"
BATCH_SIZE=5

case "$MODEL" in
  llava)    FAM=llava_1_5;  MP="$MODEL_ROOT/llava-1.5-7b-hf";     AD=results/lora/llava/adapter ;;
  qwen)     FAM=qwen2_5_vl; MP="$MODEL_ROOT/Qwen2.5-VL-7B-Instruct"; AD=results/lora/qwen/adapter ;;
  internvl) FAM=internvl3_5_original; MP="$MODEL_ROOT/InternVL3_5-8B"; AD=results/lora/internvl/adapter-original ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 1 ;;
esac
mkdir -p "results/generation/$MODEL/greedy" "results/generation/$MODEL/samples"

for dataset in vilp hallusionbench mmvet; do
  exec 9>"results/generation/$MODEL/.${dataset}.lock"
  flock 9
  case $dataset in
    vilp) SRC="$DATA_ROOT/vilp/ViLP.parquet" ;;
    hallusionbench) SRC="$DATA_ROOT/HallusionBench/data" ;;
    mmvet) SRC="$DATA_ROOT/MMVet/data/test-00000-of-00001.parquet" ;;
  esac

  echo "========================================================================"
  echo " Hugging Face greedy generation: model=$MODEL dataset=$dataset"
  echo "========================================================================"
  $PYTHON_BIN scripts/generation/generate_responses.py \
    --dataset "$dataset" \
    --dataset-source "$SRC" \
    --model-family "$FAM" \
    --model-path "$MP" \
    --adapter-path "$AD" \
    --output "results/generation/$MODEL/greedy/$dataset.jsonl" \
    --phase greedy \
    --num-samples 0 \
    --seed 42 \
    --max-new-tokens 512 \
    --max-batch-size "$BATCH_SIZE"

  echo ""
  echo "========================================================================"
  echo " Hugging Face sampled generation (K=10): model=$MODEL dataset=$dataset"
  echo "========================================================================"
  $PYTHON_BIN scripts/generation/generate_responses.py \
    --dataset "$dataset" \
    --dataset-source "$SRC" \
    --model-family "$FAM" \
    --model-path "$MP" \
    --adapter-path "$AD" \
    --output "results/generation/$MODEL/samples/$dataset.jsonl" \
    --phase samples \
    --num-samples 10 \
    --reject-resample-k 50 \
    --seed 42 \
    --max-new-tokens 512 \
    --max-batch-size "$BATCH_SIZE"
  flock -u 9
  exec 9>&-
done
