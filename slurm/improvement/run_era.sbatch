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

MODEL=${MODEL:?MODEL must be llava|qwen|internvl}
cd /home/lexiangrui/Uncertainty-Quantification-of-MLLM
mkdir -p logs/era results/era_components/"$MODEL"
shift_args=("$@")

case "$MODEL" in
  llava)    FAM=llava_1_5;  MP=/opt/lexiangrui/models/llava-1.5-7b-hf;     AD=results/lora/llava/adapter ;;
  qwen)     FAM=qwen2_5_vl; MP=/opt/lexiangrui/models/Qwen2.5-VL-7B-Instruct; AD=results/lora/qwen/adapter ;;
  internvl) FAM=internvl3_5; MP=/opt/lexiangrui/models/InternVL3_5-8B-HF; AD=results/lora/internvl/adapter ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 1 ;;
esac

for dataset in vilp hallusionbench mmvet; do
  case "$dataset" in
    vilp) SRC=/opt/lexiangrui/datasets/vilp/ViLP.parquet ;;
    hallusionbench) SRC=/opt/lexiangrui/datasets/HallusionBench/data ;;
    mmvet) SRC=/opt/lexiangrui/datasets/MMVet/data/test-00000-of-00001.parquet ;;
  esac
  GREEDY_BASE="${GREEDY_DIR:-results/generation}"
  ~/.venvs/vlm-transformers/bin/python scripts/improvement/run_era.py \
    --greedy-input "$GREEDY_BASE"/"$MODEL"/greedy/"$dataset".jsonl \
    --output results/era_components/"$MODEL"/"$dataset".jsonl \
    --family "$FAM" \
    --model-path "$MP" \
    --adapter-path "$AD" \
    --dataset-source "$SRC" \
    "${shift_args[@]}"
done
