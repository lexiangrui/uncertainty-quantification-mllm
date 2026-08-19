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

# Two-Stage Decoupled Pipeline:
# Stage 1: vLLM continuous batching generation (greedy + K=10 samples)
# Stage 2: HuggingFace single-pass forward backfill for UMPIRE hidden states
# Usage: sbatch --export=MODEL=llava,DATASET=vilp slurm/generation/generate_single.sbatch
set -euo pipefail
trap 'echo "SCRIPT FAILED at line $LINENO (exit $?)" >&2' ERR

MODEL=${MODEL:?MODEL must be llava|qwen|internvl}
DATASET=${DATASET:?DATASET must be vilp|hallusionbench|mmvet}
cd /home/lexiangrui/Uncertainty-Quantification-of-MLLM
mkdir -p logs/generation

export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="/home/lexiangrui/.local/bin:/home/lexiangrui/.venvs/vllm-0.25.1/bin:$PATH"
PYTHON_BIN=~/.venvs/vllm-0.25.1/bin/python

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

echo "========================================================================"
echo " [Stage 1.1] vLLM Greedy Generation: model=$MODEL dataset=$DATASET"
echo "========================================================================"
$PYTHON_BIN scripts/generation/generate_responses.py \
  --dataset "$DATASET" \
  --dataset-source "$SRC" \
  --model-family "$FAM" \
  --model-path "$MP" \
  --adapter-path "$AD" \
  --output "results/generation/$MODEL/greedy/$DATASET.jsonl" \
  --phase greedy \
  --num-samples 0 \
  --engine vllm \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 \
  --request-window-samples 64 \
  --seed 42 \
  --max-new-tokens 512

echo ""
echo "========================================================================"
echo " [Stage 1.2] vLLM Sampled Generation (K=10): model=$MODEL dataset=$DATASET"
echo "========================================================================"
$PYTHON_BIN scripts/generation/generate_responses.py \
  --dataset "$DATASET" \
  --dataset-source "$SRC" \
  --model-family "$FAM" \
  --model-path "$MP" \
  --adapter-path "$AD" \
  --output "results/generation/$MODEL/samples/$DATASET.jsonl" \
  --phase samples \
  --num-samples 10 \
  --reject-resample-k 50 \
  --engine vllm \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 \
  --request-window-samples 64 \
  --seed 42 \
  --max-new-tokens 512

echo ""
echo "========================================================================"
echo " [Stage 2] HF Forward Backfill (UMPIRE .pt Sidecars): model=$MODEL dataset=$DATASET"
echo "========================================================================"
$PYTHON_BIN scripts/generation/backfill_hidden_states.py \
  --dataset "$DATASET" \
  --dataset-source "$SRC" \
  --model-family "$FAM" \
  --model-path "$MP" \
  --adapter-path "$AD" \
  --samples-input "results/generation/$MODEL/samples/$DATASET.jsonl"

echo ""
echo "=== Two-Stage Generation Pipeline completed successfully for $MODEL on $DATASET ==="
