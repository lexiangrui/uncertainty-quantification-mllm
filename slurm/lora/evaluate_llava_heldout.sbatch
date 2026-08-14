#!/bin/bash
#SBATCH --job-name=llava-heldout
#SBATCH --partition=debug01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/lexiangrui/Uncertainty-Quantification-of-MLLM/logs/lora/%x_%j.out
#SBATCH --error=/home/lexiangrui/Uncertainty-Quantification-of-MLLM/logs/lora/%x_%j.err

set -euo pipefail
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
: "${ADAPTER_PATH:?set ADAPTER_PATH}"
: "${OUTPUT:?set OUTPUT}"
PYTHON_BIN="${PYTHON_BIN:-/home/lexiangrui/.venvs/vlm-transformers/bin/python}"
SRUN_BIN="${SRUN_BIN:-/opt/slurm/bin/srun}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs/lora" "$(dirname "$OUTPUT")"

REJECT_RESAMPLE_K="${REJECT_RESAMPLE_K:-50}"
TEMPERATURE="${TEMPERATURE:-1.0}"

"$SRUN_BIN" "$PYTHON_BIN" LoRA/scripts/evaluate_llava_heldout.py \
    --model-path /opt/lexiangrui/models/llava-1.5-7b-hf \
    --adapter-path "$ADAPTER_PATH" \
    --test-jsonl /opt/lexiangrui/datasets/vqav2_xml_sft/final/test.jsonl \
    --image-dir /opt/lexiangrui/datasets/vqav2_xml_sft/images \
    --output "$OUTPUT" \
    --max-new-tokens 256 \
    --num-samples 10 \
    --reject-resample-k "$REJECT_RESAMPLE_K" \
    --temperature "$TEMPERATURE" \
    --seed 42
