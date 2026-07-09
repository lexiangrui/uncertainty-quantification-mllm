#!/bin/bash
#SBATCH --job-name=vauq-paper-grid
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

if command -v module >/dev/null 2>&1; then
    module purge
fi

export PATH="/opt/slurm/bin:$PATH"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results

if [ -f configs/vauq.env ]; then
    source configs/vauq.env
else
    echo "ERROR: configs/vauq.env missing (copy from vauq.env.example)" >&2
    exit 1
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TASKS=(
    "llava|LLAVA_7B_HF_MODEL|vilp|none|core|results/vilp_llava-1.5-7b-hf_paper_v2.jsonl"
    "llava|LLAVA_7B_HF_MODEL|mmvet|none|core|results/mmvet_llava-1.5-7b-hf_paper_v2.jsonl"
    "llava|LLAVA_7B_HF_MODEL|cvbench|letter|core|results/cvbench_llava-1.5-7b-hf_paper_v2.jsonl"
    "llava|LLAVA_13B_HF_MODEL|vilp|none|core|results/vilp_llava-1.5-13b-hf_paper_v2.jsonl"
    "llava|LLAVA_13B_HF_MODEL|mmvet|none|core|results/mmvet_llava-1.5-13b-hf_paper_v2.jsonl"
    "llava|LLAVA_13B_HF_MODEL|cvbench|letter|core|results/cvbench_llava-1.5-13b-hf_paper_v2.jsonl"
)

TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r BACKEND MODEL_VAR BENCHMARK JUDGE MASK_STRATEGY OUT_PATH <<< "$TASK"
MODEL_PATH="${!MODEL_VAR}"

LIMIT_ARGS=()
if [ -n "${LIMIT:-}" ] && [ "${LIMIT}" != "0" ]; then
    LIMIT_ARGS=(--limit "$LIMIT")
fi

echo "task=${SLURM_ARRAY_TASK_ID} backend=${BACKEND} benchmark=${BENCHMARK} model=${MODEL_PATH}"
echo "output=${OUT_PATH} limit=${LIMIT:-full}"
echo "generation=greedy max_new_tokens=128 judge=${JUDGE}"

LOCK_PATH="logs/$(basename "$OUT_PATH").lock"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    echo "Another job is already writing ${OUT_PATH}; exiting to avoid duplicate writes."
    exit 0
fi

"$PYTHON_BIN" scripts/run_vauq.py \
    --backend "$BACKEND" \
    --benchmark "$BENCHMARK" \
    --judge "$JUDGE" \
    --model-path "$MODEL_PATH" \
    --mask-strategy "$MASK_STRATEGY" \
    --inference-temp 0 \
    --max-new-tokens 128 \
    --resume \
    "${LIMIT_ARGS[@]}" \
    --output "$OUT_PATH"
