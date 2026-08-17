#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/lexiangrui/Uncertainty-Quantification-of-MLLM}"
PYTHON_BIN="${PYTHON_BIN:-/home/lexiangrui/.venvs/vlm-transformers/bin/python}"
SBATCH_BIN="${SBATCH_BIN:-/opt/slurm/bin/sbatch}"
GENERATION_ROOT="$PROJECT_ROOT/results/generation"
UQ_ROOT="$PROJECT_ROOT/results/uq"
ENTAILMENT_MODEL_PATH="${ENTAILMENT_MODEL_PATH:-/opt/lexiangrui/sem_unc_assets/models/deberta-v2-xlarge-mnli}"
UQ_PARTITION="${UQ_PARTITION:-batch}"
UQ_NODELIST="${UQ_NODELIST:-}"
GPU_EXCLUDE_NODES="${GPU_EXCLUDE_NODES:-gpu03}"
SBATCH_NODE_ARGS=()
if [[ -n "$GPU_EXCLUDE_NODES" ]]; then
    SBATCH_NODE_ARGS+=("--exclude=$GPU_EXCLUDE_NODES")
fi
if [[ -n "$UQ_NODELIST" ]]; then
    SBATCH_NODE_ARGS+=("--nodelist=$UQ_NODELIST")
fi

mkdir -p "$PROJECT_ROOT/logs/generation" "$UQ_ROOT"
cd "$PROJECT_ROOT"

for model in llava qwen internvl; do
    for dataset in vilp hallusionbench mmvet; do
        greedy="$GENERATION_ROOT/$model/greedy/$dataset.jsonl"
        samples="$GENERATION_ROOT/$model/samples/$dataset.jsonl"
        output="$UQ_ROOT/$model/$dataset.jsonl"

        if [[ ! -f "$greedy" || ! -f "$samples" ]]; then
            echo "WARNING: incomplete generation inputs for $model/$dataset, skipping UQ"
            continue
        fi

        mkdir -p "$(dirname "$output")"
        job_id=$(PROJECT_ROOT="$PROJECT_ROOT" PYTHON_BIN="$PYTHON_BIN" \
            GREEDY_INPUT="$greedy" SAMPLE_INPUT="$samples" OUTPUT="$output" \
            ENTAILMENT_MODEL_PATH="$ENTAILMENT_MODEL_PATH" ENTAILMENT_DEVICE=cuda \
            ENTAILMENT_BATCH_SIZE=32 \
            "$SBATCH_BIN" --parsable --partition="$UQ_PARTITION" \
            "${SBATCH_NODE_ARGS[@]}" --gres=gpu:1 --time=00:30:00 \
            --job-name="uq-$model-$dataset" --export=ALL \
            slurm/generation/compute_uq.sbatch)
        printf '%s/%s job=%s output=%s\n' "$model" "$dataset" "$job_id" "$output"
    done
done
