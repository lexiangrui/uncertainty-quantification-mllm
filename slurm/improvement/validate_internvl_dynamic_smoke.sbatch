#!/bin/bash
#SBATCH --job-name=era-internvl-dyn-smoke
#SBATCH --partition=batch,debug01
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/era/%x-%j.out
#SBATCH --error=logs/era/%x-%j.out

set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"
cd "$PROJECT_ROOT"

LOCKED_IDS=${LOCKED_IDS:-results/analysis/luh/internvl_subset_ids_pre_dynamic_fix.txt}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/validation/internvl_dynamic_smoke}
PER_CLASS_PER_DATASET=${PER_CLASS_PER_DATASET:-1}
POSITIVE_COUNT=${POSITIVE_COUNT:-200}

mkdir -p "$OUTPUT_ROOT/era" logs/era
awk -v target="$PER_CLASS_PER_DATASET" -v positives="$POSITIVE_COUNT" '
  BEGIN { hp=vp=mp=hn=vn=mn=0 }
  NR <= positives && /^hallusionbench-/ && hp < target { print; hp++; next }
  NR <= positives && /^vilp-/ && vp < target { print; vp++; next }
  NR <= positives && /^mmvet-/ && mp < target { print; mp++; next }
  NR > positives && /^hallusionbench-/ && hn < target { print; hn++; next }
  NR > positives && /^vilp-/ && vn < target { print; vn++; next }
  NR > positives && /^mmvet-/ && mn < target { print; mn++; next }
' "$LOCKED_IDS" > "$OUTPUT_ROOT/sample_ids.txt"

expected=$((6 * PER_CLASS_PER_DATASET))
actual=$(awk 'NF { count++ } END { print count + 0 }' "$OUTPUT_ROOT/sample_ids.txt")
if [[ "$actual" -ne "$expected" ]]; then
  echo "expected $expected smoke IDs, found $actual" >&2
  exit 1
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
for dataset in vilp hallusionbench mmvet; do
  case "$dataset" in
    vilp) source_path="$DATA_ROOT/vilp/ViLP.parquet" ;;
    hallusionbench) source_path="$DATA_ROOT/HallusionBench/data" ;;
    mmvet) source_path="$DATA_ROOT/MMVet/data/test-00000-of-00001.parquet" ;;
  esac
  "$HF_PYTHON" scripts/improvement/run_era.py \
    --greedy-input "results/generation/internvl/greedy/$dataset.jsonl" \
    --output "$OUTPUT_ROOT/era/$dataset.jsonl" \
    --family internvl3_5_original \
    --model-path "$MODEL_ROOT/InternVL3_5-8B" \
    --adapter-path results/lora/internvl/adapter-original \
    --dataset-source "$source_path" \
    --sample-ids-file "$OUTPUT_ROOT/sample_ids.txt"
done
