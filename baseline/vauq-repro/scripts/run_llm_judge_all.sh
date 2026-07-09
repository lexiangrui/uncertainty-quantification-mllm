#!/bin/bash
# Apply the configured LLM judge to all free-form VAUQ result files.
# Run this on mg01 after GPU VAUQ jobs have produced JSONL outputs.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f configs/vauq.env ]; then
    source configs/vauq.env
fi
if [ -f configs/llm_judge.env ]; then
    source configs/llm_judge.env
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0}"
LIMIT_ARGS=()
if [ -n "${LIMIT:-}" ] && [ "${LIMIT}" != "0" ]; then
    LIMIT_ARGS=(--limit "$LIMIT")
fi

shopt -s nullglob
files=(results/{vilp,mmvet}_*_vauq.jsonl)
if [ "${#files[@]}" -eq 0 ]; then
    echo "No free-form VAUQ result files found under results/."
    exit 0
fi

for file in "${files[@]}"; do
    if [[ "$file" == *.llm_judged.jsonl ]]; then
        continue
    fi
    echo "LLM judging $file"
    "$PYTHON_BIN" scripts/apply_llm_judge.py \
        --input "$file" \
        --resume \
        --sleep "$SLEEP_SECONDS" \
        "${LIMIT_ARGS[@]}"
done
