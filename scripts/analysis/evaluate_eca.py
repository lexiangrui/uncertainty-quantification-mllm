#!/usr/bin/env python3
"""Evaluation of ECA on the LUH subsets.

Layer band is passed explicitly; it is specified in relative depth so it
maps to each model's layer count.  Per model the per-layer feature values
are averaged inside the band (U_ECA = mean_l U_ECA^(l)).  Bands chosen
from the LUH sweep are in-sample for the band choice — label as such.

Methods compared on identical samples and judge labels:
  PPL / SE / UMPIRE (baselines) and the uncertainty-oriented ECA features.
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
from src.improvement.eca import FEATURES, layer_features

MODELS = ("llava", "qwen", "internvl")
DATASETS = ("hallusionbench", "vilp", "mmvet")
BASELINES = ("perplexity", "semantic_entropy", "umpire")


def band_mean(feats: dict[int, dict[str, float]], band: tuple[float, float]) -> dict[str, float]:
    L = max(feats) + 1
    ls = [l for l in sorted(feats) if band[0] <= l / (L - 1) <= band[1]]
    if not ls:
        raise ValueError(f"layer band {band} selects no layer out of {L}")
    return {k: float(np.mean([feats[l][k] for l in ls])) for k in feats[ls[0]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, default=PROJECT_ROOT / "results/analysis/luh/per_model_subsets.json")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--band-rel", nargs=2, type=float, required=True, metavar=("LO", "HI"),
                        help="locked layer band as relative depth, e.g. 0.33 1.0")
    parser.add_argument("--features", nargs="+", default=list(FEATURES))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/analysis/eca/final_evaluation_v3.json")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    band = tuple(args.band_rel)
    if not 0.0 <= band[0] <= band[1] <= 1.0:
        parser.error("--band-rel must satisfy 0 <= LO <= HI <= 1")

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
    print(f"locked band (relative depth): [{band[0]}, {band[1]}]")
    print(f"{'model':10s} {'n':>4s} {'method':18s} {'AUROC':>7s} {'CI95':>17s} {'AUPRC':>7s} {'PRR':>7s}")
    for model in args.models:
        sub = subsets[model]
        subset_ids = set(sub["positive_ids"]) | set(sub["negative_ids"])
        feats, uq, judge = {}, {}, {}
        for ds in DATASETS:
            p = PROJECT_ROOT / f"results/eca_components_v3/{model}/{ds}.jsonl"
            if p.exists():
                for obj in load_jsonl_records(p):
                    if obj.get("record_type") != "sample":
                        continue
                    sid = obj.get("sample", {}).get("sample_id")
                    if sid in subset_ids:
                        feats[sid] = band_mean(layer_features(obj["eca"]), band)
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

        sids = sorted(s for s in subset_ids if s in feats and s in uq and s in judge)
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
