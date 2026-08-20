#!/usr/bin/env python3
"""Evaluate ERA scores directly on the 10-sample smoke cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation.metrics import auprc, auroc, prr
from src.improvement.era import layer_features
from src.utils import load_jsonl_records


def evaluate_cell(era_path: Path, judge_path: Path, layers: tuple[int, ...]) -> dict:
    era = {
        row["sample"]["sample_id"]: row["era"]
        for row in load_jsonl_records(era_path)
        if row.get("record_type") == "sample" and row.get("era")
    }
    labels = {
        row["sample"]["sample_id"]: int(bool(row["judge"].get("hallucination")))
        for row in load_jsonl_records(judge_path)
        if row.get("record_type") == "sample" and row.get("judge", {}).get("valid") is True
    }
    scores, targets = [], []
    for sample_id, payload in era.items():
        if sample_id not in labels:
            continue
        features = layer_features(payload)
        selected = [features[layer]["U_ERA"] for layer in layers if layer in features]
        if selected:
            scores.append(float(np.mean(selected)))
            targets.append(labels[sample_id])
    if not scores:
        return {"n": 0, "reason": "no overlapping ERA and judge records"}
    score_array = np.asarray(scores, dtype=float)
    target_array = np.asarray(targets, dtype=int)
    return {
        "n": len(scores),
        "positives": int(target_array.sum()),
        "negatives": int(len(target_array) - target_array.sum()),
        "U_ERA": {
            "auroc": auroc(score_array, target_array),
            "auprc": auprc(score_array, target_array),
            "prr": prr(score_array, target_array),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    report = {"layers": args.layers, "cells": {}}
    for model in ("llava", "qwen", "internvl"):
        for dataset in ("vilp", "hallusionbench", "mmvet"):
            era = args.results_root / "era_components" / model / f"{dataset}.jsonl"
            judge = args.results_root / "judging" / model / f"{dataset}.jsonl"
            key = f"{model}/{dataset}"
            if not era.is_file() or not judge.is_file():
                report["cells"][key] = {"n": 0, "reason": "missing input"}
            else:
                report["cells"][key] = evaluate_cell(era, judge, tuple(args.layers))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
