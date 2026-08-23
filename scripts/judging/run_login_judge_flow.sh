#!/bin/bash
# Login-node API judge flow: wait for each (model, dataset) greedy replay to
# finish, then judge it with GPT-5.6-Terra. Resumable per sample; a final sweep
# retries failures and api_error leftovers.
#
# Usage: nohup bash scripts/judging/run_login_judge_flow.sh &
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

HF_PYTHON="${HF_PYTHON:-$HOME/.venvs/vlm-transformers/bin/python}"
DATA_ROOT="${DATA_ROOT:-/opt/${USER}/datasets}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.6-terra}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-300}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-10}"
POLL_SECONDS="${POLL_SECONDS:-60}"

# Prefer the project-local .ven credentials over an inherited shell setting.
unset OPENAI_BASE_URL OPENAI_API_KEY

MODELS=(llava qwen internvl)
DATASETS=(vilp hallusionbench mmvet)

mkdir -p logs/judging
exec 9>"logs/judging/.login-judge.lock"
flock -n 9 || { echo "login judge flow already running" >&2; exit 1; }

dataset_source() {
  case "$1" in
    vilp) echo "$DATA_ROOT/vilp/ViLP.parquet" ;;
    hallusionbench) echo "$DATA_ROOT/HallusionBench/data" ;;
    mmvet) echo "$DATA_ROOT/MMVet/data/test-00000-of-00001.parquet" ;;
  esac
}

replay_ready() {
  "$HF_PYTHON" - "$1" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as handle:
        run = json.loads(handle.readline()).get("run", {})
except (OSError, json.JSONDecodeError):
    sys.exit(1)
sys.exit(0 if run.get("replay_complete") is True else 1)
PY
}

generation_running() {
  squeue -h -u "$USER" -o "%j" 2>/dev/null | grep -q '^gen+uq'
}

judge_one() {
  local model="$1" dataset="$2"
  local out="results/judging/$model/$dataset.jsonl"
  echo "=== $(date '+%F %T') judge model=$model dataset=$dataset judge=$JUDGE_MODEL ==="
  mkdir -p "results/judging/$model"
  if "$HF_PYTHON" scripts/judging/judge_responses.py \
      --dataset "$dataset" \
      --dataset-source "$(dataset_source "$dataset")" \
      --greedy-input "results/generation/$model/greedy/$dataset.jsonl" \
      --output "$out" \
      --model "$JUDGE_MODEL" \
      --max-tokens "$JUDGE_MAX_TOKENS" \
      --timeout "$JUDGE_TIMEOUT" \
      --concurrency "$JUDGE_CONCURRENCY"; then
    touch "results/judging/$model/.$dataset.done"
  else
    echo "!!! judge failed: model=$model dataset=$dataset" >&2
    return 1
  fi
}

failures=0
for pass in 1 2; do
  if ((pass == 2)); then
    echo "=== $(date '+%F %T') sweep pass: retry failures and api_error leftovers ==="
  fi
  for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      local_marker="results/judging/$model/.$dataset.done"
      if ((pass == 1)) && [[ -f "$local_marker" ]]; then
        continue
      fi
      if [[ ! -f "$local_marker" ]]; then
        greedy="results/generation/$model/greedy/$dataset.jsonl"
        while ! replay_ready "$greedy"; do
          if ! generation_running; then
            echo "!!! skip $model/$dataset: generation ended without completing $greedy" >&2
            failures=$((failures + 1))
            continue 2
          fi
          sleep "$POLL_SECONDS"
        done
      fi
      judge_one "$model" "$dataset" || failures=$((failures + 1))
    done
  done
done

echo "=== $(date '+%F %T') login-node judge flow finished (failures=$failures) ==="
exit "$((failures > 0))"
