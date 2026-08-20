#!/bin/bash
# Login-node watcher: submit the offline LoRA job only after snapshot_download
# has materialized the index and all four safetensors shards.
set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/lexiangrui/Uncertainty-Quantification-of-MLLM}
MODEL_ROOT=${MODEL_ROOT:-/opt/${USER}/models}
MODEL_DIR="$MODEL_ROOT/InternVL3_5-8B"
cd "$PROJECT_ROOT"
mkdir -p logs/lora
# ssh-launched non-login shells do not inherit the cluster's Slurm PATH.
if ! command -v squeue >/dev/null 2>&1 || ! command -v sbatch >/dev/null 2>&1; then
  export PATH="$(bash -lc 'printf %s "$PATH"')"
fi
while :; do
  ready=1
  for file in model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors model.safetensors.index.json; do
    test -s "$MODEL_DIR/$file" || ready=0
  done
  if ((ready)); then
    if ! squeue -h -u "$USER" -n internvl35-original-lora | grep -q .; then
      exec sbatch slurm/lora/train_internvl35_original.sbatch
    fi
    exit 0
  fi
  sleep 60
done
