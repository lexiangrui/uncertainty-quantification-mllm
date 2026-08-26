#!/bin/bash
# Run on a network-enabled login node. Existing Gemini labels are reused for
# XML-LoRA; only native-prompt responses are sent to the API.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

HF_PYTHON="${HF_PYTHON:-$HOME/.venvs/MLLM-UQ/bin/python}"
DATA_ROOT="${DATA_ROOT:-/opt/${USER:?}/datasets}"
JUDGE_MODEL="${JUDGE_MODEL:-gemini-3.7-flash}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-10}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-300}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"

[[ -x "$HF_PYTHON" ]] || { echo "MLLM-UQ Python is missing: $HF_PYTHON" >&2; exit 1; }
mkdir -p logs/ablation
exec 9>"logs/ablation/.xml-format-judge.lock"
flock -n 9 || { echo "XML-format judge flow is already running" >&2; exit 1; }

for model in llava qwen internvl; do
  "$HF_PYTHON" scripts/ablation/run_xml_format_judging.py \
    --tested-model "$model" \
    --data-root "$DATA_ROOT" \
    --judge-model "$JUDGE_MODEL" \
    --concurrency "$JUDGE_CONCURRENCY" \
    --timeout "$JUDGE_TIMEOUT" \
    --max-tokens "$JUDGE_MAX_TOKENS"
done

"$HF_PYTHON" scripts/analysis/xml_format_ablation.py \
  --judge-model "$JUDGE_MODEL"
