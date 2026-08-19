#!/bin/bash
#SBATCH --job-name=gen-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/generation/%x-%j.out
#SBATCH --error=logs/generation/%x-%j.out

set -euo pipefail
trap 'echo "SMOKE FAILED at line $LINENO (exit $?)" >&2' ERR
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"

cd "$PROJECT_ROOT"
mkdir -p logs/generation

export VLLM_USE_V1=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$(dirname "$VLLM_PYTHON"):$PATH"

MODEL=llava
FAMILY=llava_1_5
MODEL_PATH="$MODEL_ROOT/llava-1.5-7b-hf"
ADAPTER_PATH=results/lora/llava/adapter
DATASET=vilp
DATASET_SOURCE="$DATA_ROOT/vilp/ViLP.parquet"
SMOKE_ROOT="results/smoke/${SLURM_JOB_ID}/${MODEL}/${DATASET}"
SAMPLES_PATH="${SMOKE_ROOT}/samples.jsonl"
mkdir -p "$SMOKE_ROOT"

$VLLM_PYTHON scripts/generation/generate_responses.py \
  --dataset "$DATASET" \
  --dataset-source "$DATASET_SOURCE" \
  --model-family "$FAMILY" \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --output "$SAMPLES_PATH" \
  --phase samples \
  --num-samples 2 \
  --reject-resample-k 10 \
  --engine vllm \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 2 \
  --request-window-samples 1 \
  --seed 42 \
  --max-new-tokens 512 \
  --limit 1

$HF_PYTHON scripts/generation/backfill_hidden_states.py \
  --dataset "$DATASET" \
  --dataset-source "$DATASET_SOURCE" \
  --model-family "$FAMILY" \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --samples-input "$SAMPLES_PATH" \
  --batch-size 2 \
  --limit 1

$HF_PYTHON -c 'import json,sys; rows=[json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]; assert len(rows)==2; assert rows[0]["backfill_manifest"]["version"]=="3.0"; assert all(s["sections_valid"] for s in rows[1]["samples"]); assert rows[1]["hidden_states"]["shape"][0]==2' "$SAMPLES_PATH"

echo "SMOKE PASSED output=$SAMPLES_PATH"
