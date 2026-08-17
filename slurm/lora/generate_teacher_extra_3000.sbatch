#!/bin/bash
#SBATCH --job-name=vqav2_teacher_extra
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=/home/lexiangrui/Uncertainty-Quantification-of-MLLM/logs/lora/%x_%j.out
#SBATCH --error=/home/lexiangrui/Uncertainty-Quantification-of-MLLM/logs/lora/%x_%j.err

set -euo pipefail

repo=/home/lexiangrui/Uncertainty-Quantification-of-MLLM
data_root=/opt/lexiangrui/datasets/vqav2_xml_sft
python=/home/lexiangrui/.venvs/vlm-transformers/bin/python3.11

mkdir -p "$repo/logs/lora"
"$python" -c 'import openai'

set -a
source "$repo/LoRA/configs/qwen_teacher.env"
set +a

if [[ -n "${TEACHER_PROXY:-}" ]]; then
    export HTTP_PROXY="$TEACHER_PROXY"
    export HTTPS_PROXY="$TEACHER_PROXY"
fi

"$python" "$repo/LoRA/scripts/validate_teacher_inputs.py" \
    --candidates "$data_root/candidates_extra_3000.jsonl" \
    --image-dir "$data_root/images" \
    --expected 3000

extra_args=()
if [[ -n "${LIMIT:-}" ]]; then
    extra_args+=(--limit "$LIMIT")
fi

accepted="${ACCEPTED:-$data_root/teacher/accepted_extra_3000_v2.jsonl}"
rejected="${REJECTED:-$data_root/teacher/rejected_extra_3000_v2.jsonl}"
touch "$accepted" "$rejected"

"$python" "$repo/LoRA/scripts/generate_teacher_data.py" \
    --candidates "$data_root/candidates_extra_3000.jsonl" \
    --image-dir "$data_root/images" \
    --accepted "$accepted" \
    --rejected "$rejected" \
    --model qwen3.7-plus \
    --workers "${WORKERS:-2}" \
    "${extra_args[@]}"

wc -l "$data_root/candidates_extra_3000.jsonl" \
    "$accepted" \
    "$rejected"
