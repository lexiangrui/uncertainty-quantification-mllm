#!/bin/bash
# Login-node watcher: start judge/UQ/ERA smoke jobs once all hybrid outputs exist.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/${USER}/Uncertainty-Quantification-of-MLLM}
cd "$PROJECT_ROOT"
mkdir -p logs/setup

if ! command -v squeue >/dev/null 2>&1 || ! command -v sbatch >/dev/null 2>&1; then
  export PATH="$(bash -lc 'printf %s "$PATH"')"
fi

models=(llava qwen internvl)
datasets=(vilp hallusionbench mmvet)
while :; do
  ready=1
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      for phase in greedy samples; do
        output="results/generation/$model/$phase/$dataset.jsonl"
        if [[ ! -s "$output" ]] || (( $(wc -l < "$output") < 11 )); then
          ready=0
        fi
      done
    done
  done
  if ((ready)); then
    if squeue -h -u "$USER" -n judge-ref-smoke,uq-smoke,era-smoke,metrics-smoke,era-metrics-smoke | grep -q .; then
      exit 0
    fi
    judge=$(sbatch --parsable --partition=debug01 slurm/judging/judge_reference_smoke.sbatch)
    uq=$(sbatch --parsable --partition=debug01 slurm/uq/compute_uq_smoke.sbatch)
    era=$(sbatch --parsable --partition=debug01 slurm/improvement/run_era_smoke.sbatch)
    metrics=$(sbatch --parsable --partition=debug01 --dependency="afterok:${judge}:${uq}" \
      slurm/evaluation/compute_metrics_smoke.sbatch)
    era_metrics=$(sbatch --parsable --partition=debug01 --dependency="afterok:${judge}:${era}" \
      slurm/analysis/evaluate_era_smoke.sbatch)
    printf 'judge=%s uq=%s era=%s metrics=%s era_metrics=%s\n' \
      "$judge" "$uq" "$era" "$metrics" "$era_metrics" \
      | tee logs/setup/hybrid-smoke-jobs.out
    exit 0
  fi
  sleep 60
done
