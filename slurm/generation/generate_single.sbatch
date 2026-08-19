#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/generation/%x-%j-%N.out
#SBATCH --error=logs/generation/%x-%j-%N.out

# Two-Stage Decoupled Pipeline:
# Stage 1: vLLM continuous batching generation (greedy + K=10 samples)
# Stage 2: HuggingFace single-pass forward backfill for UMPIRE hidden states
# Usage: sbatch --export=MODEL=llava,DATASET=vilp slurm/generation/generate_single.sbatch
set -euo pipefail
trap 'echo "SCRIPT FAILED at line $LINENO (exit $?)" >&2' ERR
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"

MODEL=${MODEL:?MODEL must be llava|qwen|internvl}
DATASET=${DATASET:?DATASET must be vilp|hallusionbench|mmvet}
cd "$PROJECT_ROOT"
mkdir -p logs/generation

export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$(dirname "$VLLM_PYTHON"):$PATH"
PYTHON_BIN="$VLLM_PYTHON"
HF_PYTHON_BIN="$HF_PYTHON"

case "$MODEL" in
  llava)    FAM=llava_1_5;  MP="$MODEL_ROOT/llava-1.5-7b-hf";     AD=results/lora/llava/adapter ;;
  qwen)     FAM=qwen2_5_vl; MP="$MODEL_ROOT/Qwen2.5-VL-7B-Instruct"; AD=results/lora/qwen/adapter ;;
  internvl) FAM=internvl3_5; MP="$MODEL_ROOT/InternVL3_5-8B-HF";    AD=results/lora/internvl/adapter ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 1 ;;
esac

case "$DATASET" in
    vilp) SRC="$DATA_ROOT/vilp/ViLP.parquet" ;;
    hallusionbench) SRC="$DATA_ROOT/HallusionBench/data" ;;
    mmvet) SRC="$DATA_ROOT/MMVet/data/test-00000-of-00001.parquet" ;;
  *) echo "unknown DATASET=$DATASET" >&2; exit 1 ;;
esac

mkdir -p "results/generation/$MODEL/greedy" "results/generation/$MODEL/samples"
exec 9>"results/generation/$MODEL/.${DATASET}.lock"
flock 9

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
$HF_PYTHON_BIN scripts/generation/backfill_hidden_states.py \
  --dataset "$DATASET" \
  --dataset-source "$SRC" \
  --model-family "$FAM" \
  --model-path "$MP" \
  --adapter-path "$AD" \
  --samples-input "results/generation/$MODEL/samples/$DATASET.jsonl"

flock -u 9
exec 9>&-

echo ""
echo "=== Two-Stage Generation Pipeline completed successfully for $MODEL on $DATASET ==="
