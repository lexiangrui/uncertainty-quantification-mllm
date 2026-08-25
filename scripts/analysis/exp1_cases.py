#!/usr/bin/env python3
"""Section 3.4 helper: verify the three showcase LUH cases against current data.

Prints, for each case sample, the aligned-judge labels and the three UQ scores
with their within-cell average-rank percentiles, plus SE cluster count,
UMPIRE components, answer length and answer format.  Also checks whether the
sample belongs to the re-extracted per-model LUH subset.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.exp1_common import (  # noqa: E402
    METHODS,
    RESULTS,
    all3_valid,
    answer_format,
    cell_percentiles,
    evaluated,
    load_cell,
)

CASES = [
    ("llava", "vilp", "vilp-55-case2", "案例1：语言先验凌驾视觉"),
    ("qwen", "mmvet", "mmvet-v1_16", "案例2：视觉读数错误"),
    ("internvl", "hallusionbench", "hallusionbench-image-VS-chart-6-1-3", "案例3：答案正确但视觉观察含幻觉"),
]


def main() -> None:
    subset_path = RESULTS / "analysis" / "luh" / "per_model_subsets.json"
    subsets = json.loads(subset_path.read_text()) if subset_path.exists() else {}
    for model, dataset, sid, title in CASES:
        records = all3_valid(evaluated(load_cell(model, dataset)))
        target = next((r for r in records if r["sample_id"] == sid), None)
        print(f"== {title} ({model}/{dataset}/{sid}) ==")
        if target is None:
            print("  样本不在三方法全有效集合中，需人工核查\n")
            continue
        pcts = cell_percentiles(records)
        i = next(idx for idx, r in enumerate(records) if r["sample_id"] == sid)
        judge_bits = (
            f"correct={target['correct']}, hallucination={target['hallucination']}, "
            f"rating={target['rating']}, types={target['hallucination_types']}"
        )
        print(f"  Judge: {judge_bits}")
        for method in METHODS:
            print(
                f"  {method}: score={target['scores'][method]:.4f}, "
                f"cell_percentile={pcts[method][i] * 100:.1f}%"
            )
        print(
            f"  SE clusters={target['se_n_clusters']}, dominant_mass={target['se_dominant_mass']}, "
            f"UMPIRE volume={target['ump_semantic_volume']}, "
            f"incoherence={target['ump_incoherence_mean']}"
        )
        print(
            f"  answer='{target['answer']}' tokens={target['token_count']} "
            f"format={answer_format(target['answer'])}"
        )
        if model in subsets:
            in_pos = sid in subsets[model]["positive_ids"]
            in_neg = sid in subsets[model]["negative_ids"]
            print(f"  子集归属: positive={in_pos}, negative={in_neg}")
        print()


if __name__ == "__main__":
    sys.exit(main())
