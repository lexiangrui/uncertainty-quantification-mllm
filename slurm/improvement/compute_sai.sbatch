#!/bin/bash
#SBATCH --job-name=sai_@MODEL@_@DS@
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sai/%x-%j.out

# SAI extraction: vision logit lens reads + semantic anchor interventions
# on the teacher-forced greedy response.
# Usage: submit via slurm/improvement/submit_sai.sh [model ...]
# Env overrides: IDS_FILE (default: full LUH subset), OUT_SUFFIX,
#   INTERVENE_LAYERS, SIGMAS, K_LOCATE, MAX_OBJECTS.
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

MODEL=@MODEL@
DS=@DS@
case "$MODEL" in
  llava)    FAMILY=llava_1_5;  MODEL_PATH=/opt/lexiangrui/models/llava-1.5-7b-hf; IDS=results/analysis/luh/llava_subset_ids.txt ;;
  qwen)     FAMILY=qwen2_5_vl; MODEL_PATH=/opt/lexiangrui/models/Qwen2.5-VL-7B-Instruct; IDS=results/analysis/luh/qwen_subset_ids.txt ;;
  internvl) FAMILY=internvl3_5; MODEL_PATH=/opt/lexiangrui/models/InternVL3_5-8B-HF; IDS=results/analysis/luh/internvl_subset_ids.txt ;;
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
IDS_FILE=${IDS_FILE:-$IDS}
OUT_SUFFIX=${OUT_SUFFIX:-}
INTERVENE_LAYERS=${INTERVENE_LAYERS:-"12 20 28"}
SIGMAS=${SIGMAS:-"1.0 2.0"}
K_LOCATE=${K_LOCATE:-16}
MAX_OBJECTS=${MAX_OBJECTS:-3}
ANCHOR_MODES=${ANCHOR_MODES:-"unembed mention_state"}
LOCATE_MODES=${LOCATE_MODES:-"topk topk_large soft"}
LENS_LAYERS=${LENS_LAYERS:-}
lens_args=()
if [ -n "${LENS_LAYERS}" ]; then
  lens_args+=(--lens-layers ${LENS_LAYERS})
fi
extra_args=()
if [ -n "${IDS_FILE}" ] && [ -f "${IDS_FILE}" ]; then
  extra_args+=(--sample-ids-file "${IDS_FILE}")
fi
exec ~/.venvs/vlm-transformers/bin/python scripts/improvement/run_sai.py \
  --greedy-input "results/generation/$MODEL/greedy/$DS.jsonl" \
  --objects-input "results/sai/$MODEL/objects/$DS.jsonl" \
  --output "results/sai/$MODEL/${OUT_SUFFIX}${DS}.jsonl" \
  --family "${FAMILY}" \
  --model-path "${MODEL_PATH}" \
  --adapter-path "results/lora/$MODEL/adapter" \
  --dataset-source "${DS_SOURCE}" \
  --intervene-layers ${INTERVENE_LAYERS} \
  --sigmas ${SIGMAS} \
  --anchor-modes ${ANCHOR_MODES} \
  --locate-modes ${LOCATE_MODES} \
  --k-locate "${K_LOCATE}" \
  --max-objects "${MAX_OBJECTS}" \
  "${lens_args[@]}" \
  "${extra_args[@]}"
