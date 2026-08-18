#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/generation/%x-%j-%N.out
#SBATCH --error=logs/generation/%x-%j-%N.out

# Single unit generation: greedy (0 samples) + samples (K=10) for one (MODEL, DATASET) pair.
# Usage: sbatch --export=MODEL=llava,DATASET=vilp slurm/generation/generate_single.sbatch
set -euo pipefail
trap 'echo "SCRIPT FAILED at line $LINENO (exit $?)" >&2' ERR

MODEL=${MODEL:?MODEL must be llava|qwen|internvl}
DATASET=${DATASET:?DATASET must be vilp|hallusionbench|mmvet}
cd /home/lexiangrui/Uncertainty-Quantification-of-MLLM
mkdir -p logs/generation

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BATCH_SIZE=5
case "$MODEL" in
  llava)    FAM=llava_1_5;  MP=/opt/lexiangrui/models/llava-1.5-7b-hf;     AD=results/lora/llava/adapter ;;
  qwen)     FAM=qwen2_5_vl; MP=/opt/lexiangrui/models/Qwen2.5-VL-7B-Instruct; AD=results/lora/qwen/adapter ;;
  internvl) FAM=internvl3_5; MP=/opt/lexiangrui/models/InternVL3_5-8B-HF;    AD=results/lora/internvl/adapter ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 1 ;;
esac

case "$DATASET" in
  vilp) SRC=/opt/lexiangrui/datasets/vilp/ViLP.parquet ;;
  hallusionbench) SRC=/opt/lexiangrui/datasets/HallusionBench/data ;;
  mmvet) SRC=/opt/lexiangrui/datasets/MMVet/data/test-00000-of-00001.parquet ;;
  *) echo "unknown DATASET=$DATASET" >&2; exit 1 ;;
esac

mkdir -p "results/generation/$MODEL/greedy" "results/generation/$MODEL/samples" "results/hidden/$MODEL"

echo "=== [1/2] Running greedy generation: model=$MODEL dataset=$DATASET ==="
~/.venvs/vlm-transformers/bin/python scripts/generation/generate_responses.py \
  --dataset "$DATASET" \
  --dataset-source "$SRC" \
  --model-family "$FAM" \
  --model-path "$MP" \
  --adapter-path "$AD" \
  --output "results/generation/$MODEL/greedy/$DATASET.jsonl" \
  --phase greedy \
  --num-samples 0 \
  --seed 42 \
  --max-new-tokens 512 \
  --max-batch-size "$BATCH_SIZE"

echo "=== [2/2] Running sampled generation (K=10): model=$MODEL dataset=$DATASET ==="
~/.venvs/vlm-transformers/bin/python scripts/generation/generate_responses.py \
  --dataset "$DATASET" \
  --dataset-source "$SRC" \
  --model-family "$FAM" \
  --model-path "$MP" \
  --adapter-path "$AD" \
  --output "results/generation/$MODEL/samples/$DATASET.jsonl" \
  --phase samples \
  --num-samples 10 \
  --reject-resample-k 50 \
  --seed 42 \
  --max-new-tokens 512 \
  --max-batch-size "$BATCH_SIZE"

echo "=== Generation completed successfully for $MODEL on $DATASET ==="
