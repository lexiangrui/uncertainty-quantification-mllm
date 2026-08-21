#!/bin/bash
#SBATCH --job-name=internvl-dyn-validate
#SBATCH --partition=batch,debug01
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/era/%x-%j.out
#SBATCH --error=logs/era/%x-%j.out

set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"
cd "$PROJECT_ROOT"

SAMPLE_IDS_FILE=${SAMPLE_IDS_FILE:-results/analysis/luh/internvl_subset_ids_pre_dynamic_fix.txt}
EXPECTED_SAMPLES=${EXPECTED_SAMPLES:-400}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/validation/internvl_dynamic_full}
GENERATION_ROOT="$OUTPUT_ROOT/generation"
MODEL_GENERATION_ROOT="$GENERATION_ROOT/internvl"
UQ_ROOT="$OUTPUT_ROOT/uq"
ERA_ROOT="$OUTPUT_ROOT/era_components"

actual=$(awk 'NF { count++ } END { print count + 0 }' "$SAMPLE_IDS_FILE")
if [[ "$actual" -ne "$EXPECTED_SAMPLES" ]]; then
  echo "expected $EXPECTED_SAMPLES sample IDs, found $actual in $SAMPLE_IDS_FILE" >&2
  exit 1
fi

mkdir -p "$MODEL_GENERATION_ROOT" "$UQ_ROOT/internvl" \
  "$ERA_ROOT/internvl" "$OUTPUT_ROOT/analysis" logs/era
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "=== [1/4] exact-token HF replay with dynamic InternVL tiles ==="
"$HF_PYTHON" scripts/generation/run_hf_replay_pipeline.py \
  --model-family internvl3_5_original \
  --model-path "$MODEL_ROOT/InternVL3_5-8B" \
  --adapter-path results/lora/internvl/adapter-original \
  --data-root "$DATA_ROOT" \
  --input-root results/generation/internvl/vllm_raw \
  --output-root "$MODEL_GENERATION_ROOT" \
  --batch-size "${REPLAY_BATCH_SIZE:-5}" \
  --sample-ids-file "$SAMPLE_IDS_FILE"

echo "=== [2/4] UQ over the fixed sample IDs ==="
for dataset in vilp hallusionbench mmvet; do
  "$HF_PYTHON" scripts/uq/compute_uq.py \
    --greedy-input "$MODEL_GENERATION_ROOT/greedy/$dataset.jsonl" \
    --sample-input "$MODEL_GENERATION_ROOT/samples/$dataset.jsonl" \
    --output "$UQ_ROOT/internvl/$dataset.jsonl" \
    --entailment-model-path "$MODEL_ROOT/DeBERTa-v3-large-mnli-fever-anli-ling-wanli" \
    --entailment-batch-size 32 \
    --entailment-device cuda
done

echo "=== [3/4] ERA over the same fixed sample IDs ==="
for dataset in vilp hallusionbench mmvet; do
  case "$dataset" in
    vilp) source_path="$DATA_ROOT/vilp/ViLP.parquet" ;;
    hallusionbench) source_path="$DATA_ROOT/HallusionBench/data" ;;
    mmvet) source_path="$DATA_ROOT/MMVet/data/test-00000-of-00001.parquet" ;;
  esac
  "$HF_PYTHON" scripts/improvement/run_era.py \
    --greedy-input "$MODEL_GENERATION_ROOT/greedy/$dataset.jsonl" \
    --output "$ERA_ROOT/internvl/$dataset.jsonl" \
    --family internvl3_5_original \
    --model-path "$MODEL_ROOT/InternVL3_5-8B" \
    --adapter-path results/lora/internvl/adapter-original \
    --dataset-source "$source_path" \
    --sample-ids-file "$SAMPLE_IDS_FILE"
done

echo "=== [4/4] paired evaluation ==="
"$HF_PYTHON" scripts/analysis/evaluate_era.py \
  --models internvl \
  --subset results/analysis/luh/per_model_subsets.json \
  --components-dir "$ERA_ROOT" \
  --generation-dir "$GENERATION_ROOT" \
  --uq-dir "$UQ_ROOT" \
  --judge-dir results/judging \
  --output "$OUTPUT_ROOT/analysis/final_evaluation.json"
