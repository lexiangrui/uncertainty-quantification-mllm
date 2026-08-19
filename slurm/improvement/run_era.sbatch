#!/bin/bash
#SBATCH --job-name=era
#SBATCH --partition=batch,debug01
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/era/%x-%j.out
#SBATCH --error=logs/era/%x-%j.out

# Usage: sbatch --export=MODEL=llava slurm/improvement/run_era.sbatch [extra run_era.py args...]
set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"

MODEL=${MODEL:?MODEL must be llava|qwen|internvl}
cd "$PROJECT_ROOT"
mkdir -p logs/era results/era_components/"$MODEL"
shift_args=("$@")

case "$MODEL" in
  llava)    FAM=llava_1_5;  MP="$MODEL_ROOT/llava-1.5-7b-hf";     AD=results/lora/llava/adapter ;;
  qwen)     FAM=qwen2_5_vl; MP="$MODEL_ROOT/Qwen2.5-VL-7B-Instruct"; AD=results/lora/qwen/adapter ;;
  internvl) FAM=internvl3_5; MP="$MODEL_ROOT/InternVL3_5-8B-HF"; AD=results/lora/internvl/adapter ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 1 ;;
esac

for dataset in vilp hallusionbench mmvet; do
  exec 9>"results/era_components/$MODEL/.${dataset}.lock"
  flock 9
  case "$dataset" in
    vilp) SRC="$DATA_ROOT/vilp/ViLP.parquet" ;;
    hallusionbench) SRC="$DATA_ROOT/HallusionBench/data" ;;
    mmvet) SRC="$DATA_ROOT/MMVet/data/test-00000-of-00001.parquet" ;;
  esac
  GREEDY_BASE="${GREEDY_DIR:-results/generation}"
  "$HF_PYTHON" scripts/improvement/run_era.py \
    --greedy-input "$GREEDY_BASE"/"$MODEL"/greedy/"$dataset".jsonl \
    --output results/era_components/"$MODEL"/"$dataset".jsonl \
    --family "$FAM" \
    --model-path "$MP" \
    --adapter-path "$AD" \
    --dataset-source "$SRC" \
    "${shift_args[@]}"
  flock -u 9
  exec 9>&-
done
