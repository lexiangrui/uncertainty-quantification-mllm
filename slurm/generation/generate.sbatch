#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --partition=batch,debug01
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/generation/%x-%j.out
#SBATCH --error=logs/generation/%x-%j.out

# Two-Stage Decoupled Pipeline (Sequentially over datasets):
# Stage 1: vLLM continuous batching generation (greedy + K=10 samples)
# Stage 2: HuggingFace single-pass forward backfill for UMPIRE hidden states
#
# Usage: sbatch --export=MODEL=llava slurm/generation/generate.sbatch
set -euo pipefail
trap 'echo "SCRIPT FAILED at line $LINENO (exit $?)" >&2' ERR

MODEL=${MODEL:?MODEL must be llava|qwen|internvl}
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

mkdir -p "results/generation/$MODEL/greedy" "results/generation/$MODEL/samples" "results/hidden/$MODEL"

for dataset in vilp hallusionbench mmvet; do
  case $dataset in
    vilp) SRC=/opt/lexiangrui/datasets/vilp/ViLP.parquet ;;
    hallusionbench) SRC=/opt/lexiangrui/datasets/HallusionBench/data ;;
    mmvet) SRC=/opt/lexiangrui/datasets/MMVet/data/test-00000-of-00001.parquet ;;
  esac

  echo "========================================================================"
  echo " [Stage 1.1] vLLM Greedy Generation: model=$MODEL dataset=$dataset"
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
    --engine vllm \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 16 \
    --seed 42 \
    --max-new-tokens 512

  echo ""
  echo "========================================================================"
  echo " [Stage 1.2] vLLM Sampled Generation (K=10): model=$MODEL dataset=$dataset"
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
    --engine vllm \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 16 \
    --seed 42 \
    --max-new-tokens 512

  echo ""
  echo "========================================================================"
  echo " [Stage 2] HF Forward Backfill (UMPIRE .pt): model=$MODEL dataset=$dataset"
  echo "========================================================================"
  $PYTHON_BIN scripts/generation/backfill_hidden_states.py \
    --dataset "$dataset" \
    --dataset-source "$SRC" \
    --model-family "$FAM" \
    --model-path "$MP" \
    --adapter-path "$AD" \
    --samples-input "results/generation/$MODEL/samples/$dataset.jsonl"
done
