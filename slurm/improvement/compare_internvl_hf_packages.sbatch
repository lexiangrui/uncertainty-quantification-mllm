#!/bin/bash
#SBATCH --job-name=internvl-hf-package
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/era/%x-%j.out
#SBATCH --error=logs/era/%x-%j.out

set -euo pipefail
source "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/slurm/common.sh"
cd "$PROJECT_ROOT"

MODEL_PATH="$MODEL_ROOT/InternVL3_5-8B-HF"
ADAPTER_PATH=results/lora/internvl/adapter-original
SUBSET=results/analysis/luh/per_model_subsets.json
SUBSET_IDS=results/analysis/luh/internvl_subset_ids.txt
REFERENCE_ROOT=results/validation/internvl_dynamic_full
OUTPUT_ROOT=${OUTPUT_ROOT:-results/validation/internvl_hf_package_control}
COMPONENTS_ROOT="$OUTPUT_ROOT/era_components"
ANALYSIS_ROOT="$OUTPUT_ROOT/analysis"

mkdir -p "$COMPONENTS_ROOT/internvl" "$ANALYSIS_ROOT" logs/era
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for required in "$MODEL_PATH" "$ADAPTER_PATH" "$SUBSET" "$SUBSET_IDS"; do
  if [[ ! -e "$required" ]]; then
    echo "required control input is missing: $required" >&2
    exit 1
  fi
done

actual=$(awk 'NF { count++ } END { print count + 0 }' "$SUBSET_IDS")
if [[ "$actual" -ne 400 ]]; then
  echo "expected 400 fixed InternVL IDs, found $actual" >&2
  exit 1
fi

echo "=== [1/3] old HF package ERA on the fixed 400 ==="
for dataset in vilp hallusionbench mmvet; do
  case "$dataset" in
    vilp) source_path="$DATA_ROOT/vilp/ViLP.parquet" ;;
    hallusionbench) source_path="$DATA_ROOT/HallusionBench/data" ;;
    mmvet) source_path="$DATA_ROOT/MMVet/data/test-00000-of-00001.parquet" ;;
  esac
  "$HF_PYTHON" scripts/improvement/run_era.py \
    --greedy-input "$REFERENCE_ROOT/generation/internvl/greedy/$dataset.jsonl" \
    --output "$COMPONENTS_ROOT/internvl/$dataset.jsonl" \
    --family internvl3_5 \
    --model-path "$MODEL_PATH" \
    --adapter-path "$ADAPTER_PATH" \
    --dataset-source "$source_path" \
    --sample-ids-file "$SUBSET_IDS"
done

echo "=== [2/3] evaluate old HF package on the fixed 400 ==="
"$HF_PYTHON" scripts/analysis/evaluate_era.py \
  --models internvl \
  --subset "$SUBSET" \
  --components-dir "$COMPONENTS_ROOT" \
  --generation-dir "$REFERENCE_ROOT/generation" \
  --uq-dir "$REFERENCE_ROOT/uq" \
  --judge-dir results/judging \
  --output "$ANALYSIS_ROOT/old_hf_fixed_subset.json"

echo "=== [3/3] paired per-sample package comparison ==="
"$HF_PYTHON" - \
  "$REFERENCE_ROOT/era_components/internvl" \
  "$COMPONENTS_ROOT/internvl" \
  "$SUBSET_IDS" \
  "$ANALYSIS_ROOT/package_comparison.json" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from src.improvement.era import layer_features
from src.utils import load_jsonl_records

current_root, old_hf_root, ids_path, output_path = map(Path, sys.argv[1:])
fixed_ids = {
    value.strip()
    for value in ids_path.read_text(encoding="utf-8").splitlines()
    if value.strip()
}


def load_scores(root: Path):
    scores = {}
    visual_tokens = {}
    for dataset in ("vilp", "hallusionbench", "mmvet"):
        for row in load_jsonl_records(root / f"{dataset}.jsonl"):
            if row.get("record_type") != "sample":
                continue
            sid = row.get("sample", {}).get("sample_id")
            payload = row.get("era")
            if sid not in fixed_ids or not payload:
                continue
            features = layer_features(payload)
            selected = [features[layer]["U_ERA"] for layer in (0, 1) if layer in features]
            if selected:
                scores[sid] = float(np.mean(selected))
                visual_tokens[sid] = int(payload["n_visual_tokens"])
    return scores, visual_tokens


current, current_visual = load_scores(current_root)
old_hf, old_hf_visual = load_scores(old_hf_root)
common = sorted(fixed_ids & current.keys() & old_hf.keys())
if len(common) != len(fixed_ids):
    raise RuntimeError(
        f"paired comparison incomplete: common={len(common)} fixed={len(fixed_ids)} "
        f"missing_current={len(fixed_ids-current.keys())} "
        f"missing_old_hf={len(fixed_ids-old_hf.keys())}"
    )

x = np.asarray([current[sid] for sid in common], dtype=np.float64)
y = np.asarray([old_hf[sid] for sid in common], dtype=np.float64)
delta = y - x
visual_equal = [current_visual[sid] == old_hf_visual[sid] for sid in common]
result = {
    "n": len(common),
    "current_package": "InternVL3_5-8B original remote-code package",
    "old_hf_package": "InternVL3_5-8B-HF native Transformers package",
    "adapter": str(Path("results/lora/internvl/adapter-original")),
    "layers": [0, 1],
    "pearson": float(np.corrcoef(x, y)[0, 1]),
    "spearman": float(spearmanr(x, y).statistic),
    "mae": float(np.mean(np.abs(delta))),
    "rmse": float(np.sqrt(np.mean(delta**2))),
    "max_abs_delta": float(np.max(np.abs(delta))),
    "mean_delta_old_hf_minus_current": float(np.mean(delta)),
    "visual_token_count_equal": int(sum(visual_equal)),
    "visual_token_count_different": int(len(visual_equal) - sum(visual_equal)),
    "per_sample": [
        {
            "sample_id": sid,
            "current": current[sid],
            "old_hf": old_hf[sid],
            "delta": old_hf[sid] - current[sid],
            "current_visual_tokens": current_visual[sid],
            "old_hf_visual_tokens": old_hf_visual[sid],
        }
        for sid in common
    ],
}
output_path.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({key: value for key, value in result.items() if key != "per_sample"}, indent=2))
PY

echo "InternVL package control output=$OUTPUT_ROOT"
