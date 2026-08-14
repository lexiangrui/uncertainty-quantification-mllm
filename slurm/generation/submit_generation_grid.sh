#!/bin/bash
set -euo pipefail

: "${GENERATION_PHASE:?set GENERATION_PHASE to greedy or samples}"
if [[ "$GENERATION_PHASE" != "greedy" && "$GENERATION_PHASE" != "samples" ]]; then
    echo "GENERATION_PHASE must be greedy or samples" >&2
    exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/home/lexiangrui/Uncertainty-Quantification-of-MLLM}"
PYTHON_BIN="${PYTHON_BIN:-/home/lexiangrui/.venvs/vlm-transformers/bin/python}"
SBATCH_BIN="${SBATCH_BIN:-/opt/slurm/bin/sbatch}"
GENERATION_ROOT="$PROJECT_ROOT/results/generation"
GPU_EXCLUDE_NODES="${GPU_EXCLUDE_NODES:-gpu03}"
SBATCH_NODE_ARGS=()
if [[ -n "$GPU_EXCLUDE_NODES" ]]; then
    SBATCH_NODE_ARGS+=("--exclude=$GPU_EXCLUDE_NODES")
fi

LLAVA_MODEL=/opt/lexiangrui/models/llava-1.5-7b-hf
QWEN_MODEL=/opt/lexiangrui/models/Qwen2.5-VL-7B-Instruct
INTERNVL_MODEL=/opt/lexiangrui/models/InternVL3_5-8B-HF
ADAPTER_ROOT="$PROJECT_ROOT/results/lora"
VILP=/opt/lexiangrui/datasets/vilp/ViLP.parquet
HALLUSIONBENCH=/opt/lexiangrui/datasets/HallusionBench/data
MMVET=/opt/lexiangrui/datasets/MMVet/data/test-00000-of-00001.parquet

mkdir -p "$PROJECT_ROOT/logs/generation" "$GENERATION_ROOT"
cd "$PROJECT_ROOT"

submit() {
    local model="$1" family="$2" model_path="$3" dataset="$4" dataset_source="$5"
    local adapter_path="$ADAPTER_ROOT/$model/adapter"
    local output="$GENERATION_ROOT/$model/$GENERATION_PHASE/$dataset.jsonl"
    local num_samples=0 model_gres="gpu:1" qwen_device_map=""

    if [[ "$GENERATION_PHASE" == "samples" ]]; then
        num_samples=10
    fi
    if [[ "$family" == "qwen2_5_vl" ]]; then
        model_gres="gpu:2"
        qwen_device_map="vision_language_split"
    fi

    mkdir -p "$(dirname "$output")"
    job_id=$(PROJECT_ROOT="$PROJECT_ROOT" PYTHON_BIN="$PYTHON_BIN" \
        DATASET="$dataset" DATASET_SOURCE="$dataset_source" \
        MODEL_FAMILY="$family" MODEL_PATH="$model_path" \
        ADAPTER_PATH="$adapter_path" OUTPUT="$output" \
        GENERATION_PHASE="$GENERATION_PHASE" NUM_SAMPLES="$num_samples" \
        MAX_BATCH_SIZE=5 REQUEST_WINDOW_SAMPLES=16 REJECT_RESAMPLE_K=50 \
        ATTN_IMPLEMENTATION=flash_attention_2 QWEN_DEVICE_MAP="$qwen_device_map" \
        "$SBATCH_BIN" --parsable --partition=batch "${SBATCH_NODE_ARGS[@]}" \
        --gres="$model_gres" --time=24:00:00 \
        --job-name="$GENERATION_PHASE-$model-$dataset" --export=ALL \
        slurm/generation/generate_responses.sbatch)
    printf '%s/%s job=%s output=%s\n' "$model" "$dataset" "$job_id" "$output"
}

submit llava llava_1_5 "$LLAVA_MODEL" vilp "$VILP"
submit llava llava_1_5 "$LLAVA_MODEL" hallusionbench "$HALLUSIONBENCH"
submit llava llava_1_5 "$LLAVA_MODEL" mmvet "$MMVET"
submit qwen qwen2_5_vl "$QWEN_MODEL" vilp "$VILP"
submit qwen qwen2_5_vl "$QWEN_MODEL" hallusionbench "$HALLUSIONBENCH"
submit qwen qwen2_5_vl "$QWEN_MODEL" mmvet "$MMVET"
submit internvl internvl3_5 "$INTERNVL_MODEL" vilp "$VILP"
submit internvl internvl3_5 "$INTERNVL_MODEL" hallusionbench "$HALLUSIONBENCH"
submit internvl internvl3_5 "$INTERNVL_MODEL" mmvet "$MMVET"
