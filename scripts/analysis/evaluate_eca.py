#!/usr/bin/env python3
"""Evaluation of ECA (U_direct) on the LUH subsets.

The method averages per-layer U_direct over absolute decoder layers
(default 0 1, frozen 2026-08-18; the layer choice comes from the LUH sweep
and is therefore in-sample — labelled as such in the docs).

Methods compared on identical samples and judge labels:
  PPL / SE / UMPIRE (baselines) and ECA (U_direct @ layers).
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

from src.utils import load_jsonl_records
from src.evaluation.metrics import (
    auprc,
    auroc,
    bootstrap_summary,
    cluster_bootstrap_indices,
    prr,
)
from src.improvement.eca import DIRECT_LAYERS, FEATURES, layer_features

MODELS = ("llava", "qwen", "internvl")
DATASETS = ("hallusionbench", "vilp", "mmvet")
BASELINES = ("perplexity", "semantic_entropy", "umpire")


def layers_mean(feats: dict[int, dict[str, float]], layers: list[int]) -> dict[str, float]:
    ls = [l for l in layers if l in feats]
    if not ls:
        raise ValueError(f"layers {layers} select none out of {sorted(feats)}")
    return {k: float(np.mean([feats[l][k] for l in ls])) for k in feats[ls[0]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, default=PROJECT_ROOT / "results/analysis/luh/per_model_subsets.json")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--layers", nargs="+", type=int, default=list(DIRECT_LAYERS),
                        help="absolute decoder layers to average (default: 0 1)")
    parser.add_argument("--features", nargs="+", default=list(FEATURES))
    parser.add_argument("--components-dir", type=Path, default=PROJECT_ROOT / "results/eca_components")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/analysis/eca/final_evaluation.json")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()

    subsets = json.loads(args.subset.read_text(encoding="utf-8"))
    group_ids = {}
    for model in MODELS:
        for ds in DATASETS:
            p = PROJECT_ROOT / f"results/generation/{model}/greedy/{ds}.jsonl"
            if p.exists():
                for obj in load_jsonl_records(p):
                    if obj.get("record_type") != "sample":
                        continue
                    sid = obj.get("sample", {}).get("sample_id")
                    if sid:
                        group_id = obj.get("sample", {}).get("group_id", sid)
                        group_ids[(model, sid)] = f"{ds}:{group_id}"

    results = {}
    print(f"ECA layers: {args.layers}")
    print(f"{'model':10s} {'n':>4s} {'method':18s} {'AUROC':>7s} {'CI95':>17s} {'AUPRC':>7s} {'PRR':>7s}")
    for model in args.models:
        sub = subsets[model]
        subset_ids = set(sub["positive_ids"]) | set(sub["negative_ids"])
        feats, uq, judge = {}, {}, {}
        for ds in DATASETS:
            p = args.components_dir / f"{model}/{ds}.jsonl"
            if p.exists():
                for obj in load_jsonl_records(p):
                    if obj.get("record_type") != "sample":
                        continue
                    sid = obj.get("sample", {}).get("sample_id")
                    if sid in subset_ids:
                        feats[sid] = layers_mean(layer_features(obj["eca"]), args.layers)
            p = PROJECT_ROOT / f"results/uq/{model}/{ds}.jsonl"
            if p.exists():
                for obj in load_jsonl_records(p):
                    if obj.get("record_type") != "sample":
                        continue
                    sid = obj.get("sample", {}).get("sample_id")
                    if sid in subset_ids:
                        row = {}
                        for m in BASELINES:
                            e = obj.get("uq", {}).get(m, {})
                            if e.get("valid") is True and e.get("score") is not None:
                                row[m] = float(e["score"])
                        if len(row) == len(BASELINES):
                            uq[sid] = row
            p = PROJECT_ROOT / f"results/judging/{model}/{ds}.jsonl"
            if p.exists():
                for obj in load_jsonl_records(p):
                    if obj.get("record_type") == "run":
                        continue
                    sid = obj.get("sample", {}).get("sample_id")
                    if sid in subset_ids:
                        j = obj.get("judge", {})
                        if j.get("valid") is True:
                            judge[sid] = 1 if j.get("hallucination") else 0

        valid_ids = sorted(subset_ids & feats.keys() & uq.keys() & judge.keys())
        if not valid_ids:
            raise ValueError(f"{model}: no overlapping samples between subset, ECA, UQ, and judge")
        sids = valid_ids
        labels = np.array([judge[s] for s in sids])
        groups = np.array([str(group_ids.get((model, s), s)) for s in sids], dtype=object)
        data = {m: np.array([uq[s][m] for s in sids]) for m in BASELINES}
        for f in args.features:
            data[f] = np.array([feats[s][f] for s in sids])
        results[model] = {"n": len(sids), "n_pos": int(labels.sum()), "metrics": {}}
        replicates = cluster_bootstrap_indices(
            groups.tolist(), n_bootstrap=args.bootstrap_samples, seed=0
        )
        for m, sc in data.items():
            a = auroc(sc, labels)
            assert a is not None
            ci = bootstrap_summary(lambda idx: auroc(sc[idx], labels[idx]), replicates)
            metrics = {
                "auroc": a,
                "auroc_ci": [ci["ci_low"], ci["ci_high"]],
                "auprc": auprc(sc, labels),
                "prr": prr(sc, labels),
            }
            results[model]["metrics"][m] = metrics
            print(f"{model:10s} {len(sids):4d} {m:18s} {a:7.4f} "
                  f"[{ci['ci_low']:7.4f},{ci['ci_high']:7.4f}] "
                  f"{metrics['auprc']:7.4f} {metrics['prr']:7.4f}")
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
