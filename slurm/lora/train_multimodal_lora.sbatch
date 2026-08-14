#!/bin/bash
#SBATCH --job-name=vlm-inline-lora
#SBATCH --partition=debug01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=/home/lexiangrui/Uncertainty-Quantification-of-MLLM/logs/lora/%x_%j.out
#SBATCH --error=/home/lexiangrui/Uncertainty-Quantification-of-MLLM/logs/lora/%x_%j.err

set -euo pipefail
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
: "${CONFIG_PATH:?set CONFIG_PATH}"
PYTHON_BIN="${PYTHON_BIN:-/home/lexiangrui/.venvs/vlm-transformers/bin/python}"
extra=()
[[ -n "${MAX_TRAIN_SAMPLES:-}" ]] && extra+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
[[ -n "${MAX_VALIDATION_SAMPLES:-}" ]] && extra+=(--max-validation-samples "$MAX_VALIDATION_SAMPLES")
[[ "${RESUME_FROM_CHECKPOINT:-0}" == "1" ]] && extra+=(--resume-from-checkpoint)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$PROJECT_ROOT/logs/lora"
cd "$PROJECT_ROOT"
child=""
forward_signal() {
    [[ -n "$child" ]] && kill -USR1 "$child" 2>/dev/null || true
}
trap forward_signal USR1
srun "$PYTHON_BIN" LoRA/scripts/train_multimodal_lora.py --config "$CONFIG_PATH" "${extra[@]}" &
child=$!
wait "$child"
