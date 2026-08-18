#!/usr/bin/env python3
"""Ablation study: comparing ECA U_direct (with A->A in denominator) vs U_direct_no_aa (without A->A).

U_direct:
    (aAV + aAR) / (aAI + aAQ + aAV + aAR + aAA + eps)
    Measures proportion of total past attention directed to self-generated context.

U_direct_no_aa:
    (aAV + aAR) / (aAI + aAQ + aAV + aAR + eps)
    Measures proportion of external prior context attention directed to self-generated context vs ground-truth inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import auprc, auroc, prr
from src.improvement.eca import DIRECT_LAYERS, layer_features
from src.utils import load_jsonl_records

MODELS = ("llava", "qwen", "internvl")
DATASETS = ("hallusionbench", "vilp", "mmvet")


def load_labels(model: str, ids: set[str]) -> dict[str, int]:
    judge = {}
    for dataset in DATASETS:
        path = PROJECT_ROOT / f"results/judging/{model}/{dataset}.jsonl"
        if not path.exists():
            continue
        for obj in load_jsonl_records(path):
            sid = obj.get("sample", {}).get("sample_id")
            if sid in ids and obj.get("judge", {}).get("valid") is True:
                judge[sid] = int(bool(obj["judge"]["hallucination"]))
    return judge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        type=Path,
        default=PROJECT_ROOT / "results/analysis/luh/per_model_subsets.json",
    )
    parser.add_argument(
        "--components-dir",
        type=Path,
        default=PROJECT_ROOT / "results/eca_components",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/analysis/eca/denominator_ablation.json",
    )
    args = parser.parse_args()

    subsets = json.loads(args.subset.read_text(encoding="utf-8"))
    results = {"with_aa (U_direct)": {}, "no_aa (U_direct_no_aa)": {}}

    print(f"{'model':10s} {'feature':22s} {'AUROC':>7s} {'AUPRC':>7s} {'PRR':>7s}")
    for model in MODELS:
        ids = set(subsets[model]["positive_ids"]) | set(subsets[model]["negative_ids"])
        components = {}
        for dataset in DATASETS:
            path = args.components_dir / f"{model}/{dataset}.jsonl"
            if not path.exists():
                continue
            for obj in load_jsonl_records(path):
                sid = obj.get("sample", {}).get("sample_id")
                if obj.get("record_type") == "sample" and sid in ids:
                    feats = layer_features(obj["eca"])
                    components[sid] = {
                        feat: float(np.mean([feats[layer][feat] for layer in DIRECT_LAYERS]))
                        for feat in ("U_direct", "U_direct_no_aa")
                    }
        labels = load_labels(model, ids)
        common_ids = sorted(ids & components.keys() & labels.keys())
        if not common_ids:
            raise ValueError(f"{model}: no overlapping samples between subset, ECA and judge")
        sids = common_ids
        target = np.array([labels[sid] for sid in sids], dtype=int)

        for feat_key, label_name in (
            ("U_direct", "with_aa (U_direct)"),
            ("U_direct_no_aa", "no_aa (U_direct_no_aa)"),
        ):
            scores = np.array([components[sid][feat_key] for sid in sids])
            v_auroc = auroc(scores, target)
            v_auprc = auprc(scores, target)
            v_prr = prr(scores, target)
            if v_auroc is None:
                raise ValueError(f"{model}: AUROC is undefined")
            results[label_name][model] = {
                "auroc": v_auroc,
                "auprc": v_auprc,
                "prr": v_prr,
            }
            print(f"{model:10s} {label_name:22s} {v_auroc:7.4f} {v_auprc:7.4f} {v_prr:7.4f}")
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
