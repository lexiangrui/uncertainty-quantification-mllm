#!/bin/bash
# vLLM token generation + HF teacher-forcing replay, limited to 10 examples.
# Usage: sbatch --export=MODEL=llava slurm/generation/generate_hybrid_smoke.sbatch
# MODEL=internvl uses the original OpenGVLab checkpoint and adapter-original.
#SBATCH --job-name=gen-hybrid-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/generation/%x-%j.out
#SBATCH --error=logs/generation/%x-%j.err

set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"
MODEL=${MODEL:?MODEL must be llava|qwen|internvl}
VLLM_PYTHON=${VLLM_PYTHON:-/home/${USER}/.venvs/vllm-0.25.1/bin/python}
cd "$PROJECT_ROOT"
mkdir -p logs/generation
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

case "$MODEL" in
  llava) FAM=llava_1_5; MP="$MODEL_ROOT/llava-1.5-7b-hf"; AD="results/lora/llava/adapter" ;;
  qwen) FAM=qwen2_5_vl; MP="$MODEL_ROOT/Qwen2.5-VL-7B-Instruct"; AD="results/lora/qwen/adapter" ;;
  internvl) FAM=internvl3_5_original; MP="$MODEL_ROOT/InternVL3_5-8B"; AD="results/lora/internvl/adapter-original" ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 1 ;;
esac

for dataset in vilp hallusionbench mmvet; do
  case "$dataset" in
    vilp) SRC="$DATA_ROOT/vilp/ViLP.parquet" ;;
    hallusionbench) SRC="$DATA_ROOT/HallusionBench/data" ;;
    mmvet) SRC="$DATA_ROOT/MMVet/data/test-00000-of-00001.parquet" ;;
  esac
  raw_root="results/generation/$MODEL/vllm_raw"
  mkdir -p "$raw_root/greedy" "$raw_root/samples" "results/generation/$MODEL/greedy" "results/generation/$MODEL/samples"

  "$VLLM_PYTHON" scripts/generation/generate_responses.py \
    --dataset "$dataset" --dataset-source "$SRC" --model-family "$FAM" \
    --model-path "$MP" --adapter-path "$AD" \
    --output "$raw_root/greedy/$dataset.jsonl" --phase greedy --num-samples 0 \
    --limit 10 --seed 42 --max-new-tokens 512 --max-batch-size 8 \
    --request-window-samples 10 --engine vllm --max-num-seqs 8 \
    --gpu-memory-utilization 0.85 --max-model-len 4096
  "$HF_PYTHON" scripts/generation/replay_hf_artifacts.py \
    --input "$raw_root/greedy/$dataset.jsonl" \
    --output "results/generation/$MODEL/greedy/$dataset.jsonl" \
    --dataset-source "$SRC" --model-family "$FAM" --model-path "$MP" \
    --adapter-path "$AD" --limit 10

  "$VLLM_PYTHON" scripts/generation/generate_responses.py \
    --dataset "$dataset" --dataset-source "$SRC" --model-family "$FAM" \
    --model-path "$MP" --adapter-path "$AD" \
    --output "$raw_root/samples/$dataset.jsonl" --phase samples --num-samples 10 \
    --limit 10 --seed 42 --max-new-tokens 512 --max-batch-size 8 \
    --request-window-samples 10 --engine vllm --max-num-seqs 8 \
    --gpu-memory-utilization 0.85 --max-model-len 4096
  "$HF_PYTHON" scripts/generation/replay_hf_artifacts.py \
    --input "$raw_root/samples/$dataset.jsonl" \
    --output "results/generation/$MODEL/samples/$dataset.jsonl" \
    --dataset-source "$SRC" --model-family "$FAM" --model-path "$MP" \
    --adapter-path "$AD" --limit 10
done
