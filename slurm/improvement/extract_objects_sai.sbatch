#!/bin/bash
#SBATCH --job-name=sai_obj_@MODEL@_@DS@
#SBATCH --partition=debug01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/sai/%x-%j.out

# SAI object extraction: independent text model (Qwen3-4B) reads the greedy
# response and lists claimed-visible visual objects with mention spans.
set -euo pipefail

export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL=@MODEL@
DS=@DS@
case "$MODEL" in
  llava)    IDS=results/analysis/luh/llava_subset_ids.txt ;;
  qwen)     IDS=results/analysis/luh/qwen_subset_ids.txt ;;
  internvl) IDS=results/analysis/luh/internvl_subset_ids.txt ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
case "$DS" in
  hallusionbench) DS_SOURCE=/opt/lexiangrui/datasets/HallusionBench/data ;;
  vilp)           DS_SOURCE=/opt/lexiangrui/datasets/vilp/ViLP.parquet ;;
  mmvet)          DS_SOURCE=/opt/lexiangrui/datasets/MMVet/data/test-00000-of-00001.parquet ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac

cd /home/lexiangrui/uq_improvement_sai
mkdir -p logs/sai
exec ~/.venvs/vlm-transformers/bin/python scripts/improvement/extract_objects_sai.py \
  --greedy-input "results/generation/$MODEL/greedy/$DS.jsonl" \
  --output "results/sai/$MODEL/objects/$DS.jsonl" \
  --sample-ids-file "$IDS" \
  "$@"
