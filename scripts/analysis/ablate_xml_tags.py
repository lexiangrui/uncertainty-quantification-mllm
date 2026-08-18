#!/usr/bin/env python3
"""XML-tag keep-vs-remove ablation for ECA U_direct."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import auroc
from src.improvement.eca import DIRECT_LAYERS, EPS
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
            if (
                sid in ids
                and obj.get("judge", {}).get("valid") is True
            ):
                judge[sid] = int(bool(obj["judge"]["hallucination"]))
    return judge


def direct_scores(
    components: dict[str, dict], sids: list[str], *, keep_tags: bool
) -> np.ndarray:
    scores = []
    for sid in sids:
        result = components[sid]
        heads = result["n_heads"]
        answer_tokens = result["section_tokens"]["answer"]
        per_layer = []
        for layer in DIRECT_LAYERS:
            mass = result["layer_masses"][str(layer)][2]
            attention = [value / (heads * answer_tokens) for value in mass]
            numerator = attention[2] + attention[3]
            denominator = sum(attention[:5])
            if keep_tags:
                numerator += attention[5]
                denominator += attention[5]
            per_layer.append(numerator / (denominator + EPS))
        scores.append(float(np.mean(per_layer)))
    return np.array(scores)


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
        default=PROJECT_ROOT / "results/analysis/eca/xml_tag_ablation.json",
    )
    args = parser.parse_args()

    subsets = json.loads(args.subset.read_text(encoding="utf-8"))
    results = {"remove_tags": {}, "keep_tags": {}}
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
                    components[sid] = obj["eca"]
        labels = load_labels(model, ids)
        missing_components = ids - components.keys()
        missing_labels = ids - labels.keys()
        if missing_components or missing_labels:
            raise ValueError(
                f"{model}: incomplete ablation inputs "
                f"(ECA missing={len(missing_components)}, "
                f"judge missing={len(missing_labels)})"
            )
        sids = sorted(ids)
        target = np.array([labels[sid] for sid in sids], dtype=int)
        for name, keep_tags in (("remove_tags", False), ("keep_tags", True)):
            value = auroc(direct_scores(components, sids, keep_tags=keep_tags), target)
            if value is None:
                raise ValueError(f"{model}: AUROC is undefined")
            results[name][model] = value
            print(f"{model:10s} {name:12s} {value:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
