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
COMPONENTS_ROOT="${ERA_COMPONENTS_DIR:-results/era_components}"
mkdir -p logs/era "$COMPONENTS_ROOT/$MODEL"
shift_args=("$@")
SUBSET_IDS_FILE="${ERA_SAMPLE_IDS_FILE:-results/analysis/luh/${MODEL}_subset_ids.txt}"
if [[ ! -s "$SUBSET_IDS_FILE" ]]; then
  echo "ERA subset ID file missing or empty: $SUBSET_IDS_FILE" >&2
  echo "Run scripts/extract_per_model_subset.py before submitting production ERA." >&2
  exit 1
fi
SUBSET_COUNT=$(awk 'NF { count += 1 } END { print count + 0 }' "$SUBSET_IDS_FILE")
EXPECTED_SUBSET_SIZE=${ERA_EXPECTED_SUBSET_SIZE:-400}
if [[ "$SUBSET_COUNT" -ne "$EXPECTED_SUBSET_SIZE" ]]; then
  echo "ERA subset must contain $EXPECTED_SUBSET_SIZE non-empty IDs; found $SUBSET_COUNT" >&2
  exit 1
fi
echo "ERA production subset: $SUBSET_IDS_FILE ($SUBSET_COUNT IDs)"

case "$MODEL" in
  llava)    FAM=llava_1_5;  MP="$MODEL_ROOT/llava-1.5-7b-hf";     AD=results/lora/llava/adapter ;;
  qwen)     FAM=qwen2_5_vl; MP="$MODEL_ROOT/Qwen2.5-VL-7B-Instruct"; AD=results/lora/qwen/adapter ;;
  internvl) FAM=internvl3_5_original; MP="$MODEL_ROOT/InternVL3_5-8B"; AD=results/lora/internvl/adapter-original ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 1 ;;
esac

ERA_ARGS=()
lock_fd=7
for dataset in vilp hallusionbench mmvet; do
  eval "exec ${lock_fd}>\"$COMPONENTS_ROOT/$MODEL/.${dataset}.lock\""
  flock "$lock_fd"
  case "$dataset" in
    vilp) SRC="$DATA_ROOT/vilp/ViLP.parquet" ;;
    hallusionbench) SRC="$DATA_ROOT/HallusionBench/data" ;;
    mmvet) SRC="$DATA_ROOT/MMVet/data/test-00000-of-00001.parquet" ;;
  esac
  GREEDY_BASE="${GREEDY_DIR:-results/generation}"
  ERA_ARGS+=(
    --task
    "$GREEDY_BASE/$MODEL/greedy/$dataset.jsonl"
    "$COMPONENTS_ROOT/$MODEL/$dataset.jsonl"
    "$SRC"
  )
  lock_fd=$((lock_fd + 1))
done

"$HF_PYTHON" scripts/improvement/run_era.py \
  --family "$FAM" \
  --model-path "$MP" \
  --adapter-path "$AD" \
  --sample-ids-file "$SUBSET_IDS_FILE" \
  "${ERA_ARGS[@]}" \
  "${shift_args[@]}"

for fd in 7 8 9; do
  flock -u "$fd"
  eval "exec ${fd}>&-"
done
