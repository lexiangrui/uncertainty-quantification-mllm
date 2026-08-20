#!/bin/bash
# Login-node watcher: submit the InternVL hybrid smoke test after LoRA training.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/${USER}/Uncertainty-Quantification-of-MLLM}
ADAPTER=${ADAPTER:-$PROJECT_ROOT/results/lora/internvl/adapter-original}
INTERNVL_JOB_NAME=${INTERNVL_JOB_NAME:-gen-hybrid-smoke-internvl}
cd "$PROJECT_ROOT"
mkdir -p logs/setup

# ssh-launched non-login shells do not inherit the cluster's Slurm PATH.
if ! command -v squeue >/dev/null 2>&1 || ! command -v sbatch >/dev/null 2>&1; then
  export PATH="$(bash -lc 'printf %s "$PATH"')"
fi

while :; do
  ready=1
  for file in adapter_config.json adapter_model.safetensors; do
    test -s "$ADAPTER/$file" || ready=0
  done
  if ((ready)); then
    outputs_ready=1
    for dataset in vilp hallusionbench mmvet; do
      for phase in greedy samples; do
        output="results/generation/internvl/$phase/$dataset.jsonl"
        if [[ ! -s "$output" ]] || (( $(wc -l < "$output") < 11 )); then
          outputs_ready=0
        fi
      done
    done
    if ((outputs_ready)); then
      exit 0
    fi
    if ! squeue -h -u "$USER" -n "$INTERNVL_JOB_NAME" | grep -q .; then
      sbatch --job-name="$INTERNVL_JOB_NAME" --partition=debug01 --export=ALL,MODEL=internvl \
        slurm/generation/generate_hybrid_smoke.sbatch
    fi
    exit 0
  fi
  sleep 60
done
