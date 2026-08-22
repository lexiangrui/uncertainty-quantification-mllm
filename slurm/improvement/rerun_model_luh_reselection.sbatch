#!/bin/bash
#SBATCH --job-name=model-luh-rerun
#SBATCH --partition=batch
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

MODEL=${MODEL:?MODEL must be llava|qwen}
case "$MODEL" in
  llava)
    FAMILY=llava_1_5
    MODEL_PATH="$MODEL_ROOT/llava-1.5-7b-hf"
    ADAPTER_PATH=results/lora/llava/adapter
    ;;
  qwen)
    FAMILY=qwen2_5_vl
    MODEL_PATH="$MODEL_ROOT/Qwen2.5-VL-7B-Instruct"
    ADAPTER_PATH=results/lora/qwen/adapter
    ;;
  *)
    echo "MODEL must be llava or qwen; found $MODEL" >&2
    exit 1
    ;;
esac

OUTPUT_ROOT=${OUTPUT_ROOT:-results/validation/${MODEL}_luh_reselection}
GENERATION_ROOT="$OUTPUT_ROOT/generation"
MODEL_GENERATION_ROOT="$GENERATION_ROOT/$MODEL"
UQ_ROOT="$OUTPUT_ROOT/uq"
ERA_ROOT="$OUTPUT_ROOT/era_components"
ANALYSIS_ROOT="$OUTPUT_ROOT/analysis"
NEW_SUBSET="$ANALYSIS_ROOT/luh/per_model_subsets.json"
NEW_IDS="$ANALYSIS_ROOT/luh/${MODEL}_subset_ids.txt"
OLD_SUBSET=results/analysis/luh/per_model_subsets.json

mkdir -p "$MODEL_GENERATION_ROOT" "$UQ_ROOT/$MODEL" \
  "$ERA_ROOT/$MODEL" "$ANALYSIS_ROOT/luh" logs/era
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [1/6] full $MODEL HF replay ==="
"$HF_PYTHON" scripts/generation/run_hf_replay_pipeline.py \
  --model-family "$FAMILY" \
  --model-path "$MODEL_PATH" \
  --adapter-path "$ADAPTER_PATH" \
  --data-root "$DATA_ROOT" \
  --input-root "results/generation/$MODEL/vllm_raw" \
  --output-root "$MODEL_GENERATION_ROOT" \
  --batch-size "${REPLAY_BATCH_SIZE:-5}"

echo "=== [2/6] full $MODEL UQ ==="
for dataset in vilp hallusionbench mmvet; do
  "$HF_PYTHON" scripts/uq/compute_uq.py \
    --greedy-input "$MODEL_GENERATION_ROOT/greedy/$dataset.jsonl" \
    --sample-input "$MODEL_GENERATION_ROOT/samples/$dataset.jsonl" \
    --output "$UQ_ROOT/$MODEL/$dataset.jsonl" \
    --entailment-model-path "$MODEL_ROOT/DeBERTa-v3-large-mnli-fever-anli-ling-wanli" \
    --entailment-batch-size 32 \
    --entailment-device cuda
done

echo "=== [3/6] reselect $MODEL LUH subset from rerun UQ ==="
"$HF_PYTHON" scripts/extract_per_model_subset.py \
  --uq-dir "$UQ_ROOT" \
  --judge-dir results/judging \
  --gen-dir "$GENERATION_ROOT" \
  --models "$MODEL" \
  --target-size 200 \
  --output "$NEW_SUBSET"

actual=$(awk 'NF { count++ } END { print count + 0 }' "$NEW_IDS")
if [[ "$actual" -ne 400 ]]; then
  echo "expected 400 reselected $MODEL IDs, found $actual" >&2
  exit 1
fi

echo "=== [4/6] rerun ERA on the reselected 400 ==="
for dataset in vilp hallusionbench mmvet; do
  case "$dataset" in
    vilp) source_path="$DATA_ROOT/vilp/ViLP.parquet" ;;
    hallusionbench) source_path="$DATA_ROOT/HallusionBench/data" ;;
    mmvet) source_path="$DATA_ROOT/MMVet/data/test-00000-of-00001.parquet" ;;
  esac
  "$HF_PYTHON" scripts/improvement/run_era.py \
    --greedy-input "$MODEL_GENERATION_ROOT/greedy/$dataset.jsonl" \
    --output "$ERA_ROOT/$MODEL/$dataset.jsonl" \
    --family "$FAMILY" \
    --model-path "$MODEL_PATH" \
    --adapter-path "$ADAPTER_PATH" \
    --dataset-source "$source_path" \
    --sample-ids-file "$NEW_IDS"
done

echo "=== [5/6] evaluate legacy and rerun cells ==="
"$HF_PYTHON" scripts/analysis/evaluate_era.py \
  --models "$MODEL" \
  --subset "$OLD_SUBSET" \
  --components-dir results/era_components \
  --generation-dir results/generation \
  --uq-dir results/uq \
  --judge-dir results/judging \
  --output "$ANALYSIS_ROOT/old_subset_legacy.json"

"$HF_PYTHON" scripts/analysis/evaluate_era.py \
  --models "$MODEL" \
  --subset "$NEW_SUBSET" \
  --components-dir "$ERA_ROOT" \
  --generation-dir "$GENERATION_ROOT" \
  --uq-dir "$UQ_ROOT" \
  --judge-dir results/judging \
  --output "$ANALYSIS_ROOT/new_subset_rerun.json"

echo "=== [6/6] record subset overlap ==="
"$HF_PYTHON" - "$MODEL" "$OLD_SUBSET" "$NEW_SUBSET" "$ANALYSIS_ROOT/subset_overlap.json" <<'PY'
import json
import sys
from pathlib import Path

model = sys.argv[1]
old_path, new_path, output_path = map(Path, sys.argv[2:])
old = json.loads(old_path.read_text(encoding="utf-8"))[model]
new = json.loads(new_path.read_text(encoding="utf-8"))[model]
result = {}
for key in ("positive_ids", "negative_ids"):
    old_ids, new_ids = set(old[key]), set(new[key])
    result[key] = {
        "old": len(old_ids),
        "new": len(new_ids),
        "overlap": len(old_ids & new_ids),
        "old_only": sorted(old_ids - new_ids),
        "new_only": sorted(new_ids - old_ids),
    }
output_path.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({key: value["overlap"] for key, value in result.items()}))
PY

echo "$MODEL rerun output=$OUTPUT_ROOT"
