#!/bin/bash
#SBATCH --job-name=gen-hybrid
#SBATCH --partition=batch,debug01
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/generation/%x-%j.out
#SBATCH --error=logs/generation/%x-%j.out

# Production path: one persistent vLLM engine generates exact tokens, then one
# persistent HF model teacher-forces those tokens in VRAM-aware batches.
set -euo pipefail
trap 'echo "SCRIPT FAILED at line $LINENO (exit $?)" >&2' ERR
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"

MODEL=${MODEL:?MODEL must be llava|qwen|internvl}
cd "$PROJECT_ROOT"
mkdir -p logs/generation
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

case "$MODEL" in
  llava) FAM=llava_1_5; MP="$MODEL_ROOT/llava-1.5-7b-hf"; AD="results/lora/llava/adapter" ;;
  qwen) FAM=qwen2_5_vl; MP="$MODEL_ROOT/Qwen2.5-VL-7B-Instruct"; AD="results/lora/qwen/adapter" ;;
  internvl) FAM=internvl3_5_original; MP="$MODEL_ROOT/InternVL3_5-8B"; AD="results/lora/internvl/adapter-original" ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 1 ;;
esac

OUTPUT_ROOT="results/generation/$MODEL"
mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.hybrid.lock"
flock 9

"$HF_PYTHON" scripts/generation/run_vllm_pipeline.py \
  --model-family "$FAM" --model-path "$MP" --adapter-path "$AD" \
  --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT"

"$HF_PYTHON" scripts/generation/run_hf_replay_pipeline.py \
  --model-family "$FAM" --model-path "$MP" --adapter-path "$AD" \
  --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT"

echo "vLLM + HF production generation completed for $MODEL"
