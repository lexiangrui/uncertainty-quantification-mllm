#!/bin/bash
# CPU validation of judge + metric code against the preserved pre-clear results.
# This does not write to results/ or alter results_backup_*.
#SBATCH --job-name=backup-metrics-validate
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/evaluation/%x-%j.out
#SBATCH --error=logs/evaluation/%x-%j.err

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-/home/${USER}/Uncertainty-Quantification-of-MLLM}
BACKUP_ROOT=${BACKUP_ROOT:-/home/${USER}/Uncertainty-Quantification-of-MLLM/results_backup_20260820_183356}
WORK_ROOT=${WORK_ROOT:-/home/${USER}/metric_validation_backup_20260820}
PYTHON=${PYTHON:-/home/${USER}/.venvs/vlm-transformers/bin/python}
DATA_ROOT=${DATA_ROOT:-/opt/${USER}/datasets}

cd "$PROJECT_ROOT"
mkdir -p logs/evaluation "$WORK_ROOT/judging" "$WORK_ROOT/metrics"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

for model in llava qwen internvl; do
  for dataset in vilp hallusionbench mmvet; do
    case "$dataset" in
      vilp) source="$DATA_ROOT/vilp/ViLP.parquet" ;;
      hallusionbench) source="$DATA_ROOT/HallusionBench/data" ;;
      mmvet) source="$DATA_ROOT/MMVet/data/test-00000-of-00001.parquet" ;;
    esac
    mkdir -p "$WORK_ROOT/judging/$model" "$WORK_ROOT/metrics/$model"
    "$PYTHON" scripts/judging/judge_reference_rule.py \
      --dataset "$dataset" --dataset-source "$source" --limit 10 \
      --greedy-input "$BACKUP_ROOT/generation/$model/greedy/$dataset.jsonl" \
      --output "$WORK_ROOT/judging/$model/$dataset.jsonl"
  done
done

# Judge outputs are written above; compute_metrics is intentionally separate so
# one malformed pair is reported by Slurm rather than silently skipped.
for model in llava qwen internvl; do
  for dataset in vilp hallusionbench mmvet; do
    "$PYTHON" scripts/evaluation/compute_metrics.py \
      --uq-input "$BACKUP_ROOT/uq/$model/$dataset.jsonl" \
      --judge-input "$WORK_ROOT/judging/$model/$dataset.jsonl" \
      --output "$WORK_ROOT/metrics/$model/$dataset.json" \
      --bootstrap-samples 100 --bootstrap-seed 0
  done
done

echo "backup_metric_validation_complete root=$WORK_ROOT"
