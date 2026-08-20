#!/bin/bash
# Compute ERA attention masses on the 3x3x10 greedy smoke outputs.
#SBATCH --job-name=era-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/era/%x-%j.out
#SBATCH --error=logs/era/%x-%j.err

set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"
cd "$PROJECT_ROOT"
mkdir -p logs/era
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
for model in llava qwen internvl; do
  case "$model" in
    llava) FAM=llava_1_5; MP="$MODEL_ROOT/llava-1.5-7b-hf"; AD="results/lora/llava/adapter" ;;
    qwen) FAM=qwen2_5_vl; MP="$MODEL_ROOT/Qwen2.5-VL-7B-Instruct"; AD="results/lora/qwen/adapter" ;;
    internvl) FAM=internvl3_5_original; MP="$MODEL_ROOT/InternVL3_5-8B"; AD="results/lora/internvl/adapter-original" ;;
  esac
  for dataset in vilp hallusionbench mmvet; do
    case "$dataset" in
      vilp) SRC="$DATA_ROOT/vilp/ViLP.parquet" ;;
      hallusionbench) SRC="$DATA_ROOT/HallusionBench/data" ;;
      mmvet) SRC="$DATA_ROOT/MMVet/data/test-00000-of-00001.parquet" ;;
    esac
    "$HF_PYTHON" scripts/improvement/run_era.py \
      --greedy-input "results/generation/$model/greedy/$dataset.jsonl" \
      --output "results/era/$model/$dataset.jsonl" --family "$FAM" \
      --model-path "$MP" --adapter-path "$AD" --dataset-source "$SRC" \
      --attn-implementation eager --limit 10
  done
done
