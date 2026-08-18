#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/generation/%x-%j.out
#SBATCH --error=logs/generation/%x-%j.out

# Full main-experiment generation: greedy + samples (K=10) phases for one model
# across all datasets, including token-ID sidecars (.tokens) and hidden states (.pt).
# Resumable via completed_sample_ids.
#
# Usage: sbatch --export=MODEL=llava slurm/generation/generate.sbatch
set -euo pipefail
trap 'echo "SCRIPT FAILED at line $LINENO (exit $?)" >&2' ERR

MODEL=${MODEL:?MODEL must be llava|qwen|internvl}
cd /home/lexiangrui/Uncertainty-Quantification-of-MLLM
mkdir -p logs/generation

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BATCH_SIZE=5
case "$MODEL" in
  llava)    FAM=llava_1_5;  MP=/opt/lexiangrui/models/llava-1.5-7b-hf;     AD=results/lora/llava/adapter ;;
  qwen)     FAM=qwen2_5_vl; MP=/opt/lexiangrui/models/Qwen2.5-VL-7B-Instruct; AD=results/lora/qwen/adapter; BATCH_SIZE=2 ;;
  internvl) FAM=internvl3_5; MP=/opt/lexiangrui/models/InternVL3_5-8B-HF;    AD=results/lora/internvl/adapter ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 1 ;;
esac

mkdir -p "results/generation/$MODEL/greedy" "results/generation/$MODEL/samples" "results/hidden/$MODEL"

for dataset in vilp hallusionbench mmvet; do
  case $dataset in
    vilp) SRC=/opt/lexiangrui/datasets/vilp/ViLP.parquet ;;
    hallusionbench) SRC=/opt/lexiangrui/datasets/HallusionBench/data ;;
    mmvet) SRC=/opt/lexiangrui/datasets/MMVet/data/test-00000-of-00001.parquet ;;
  esac

  echo "=== Running greedy generation: model=$MODEL dataset=$dataset ==="
  ~/.venvs/vlm-transformers/bin/python scripts/generation/generate_responses.py \
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

  echo "=== Running sampled generation (K=10): model=$MODEL dataset=$dataset ==="
  ~/.venvs/vlm-transformers/bin/python scripts/generation/generate_responses.py \
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
done
