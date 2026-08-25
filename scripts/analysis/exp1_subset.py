#!/usr/bin/env python3
"""Section 3.5: low-uncertainty difficult-subset construction and validation.

Re-extracts the per-model 200-positive / 200-negative subsets with the exact
production procedure (``scripts.extract_per_model_subset.extract_per_model``:
pooled over the three datasets, judge-valid + all-three-scores-valid + has
image; positives = 200 hallucinated samples with the lowest average
percentile across methods; negatives = greedy one-to-one nearest neighbours
of the positives in the 3-D per-method percentile space).  Then reports
composition, matching quality and baseline detection performance on the
subset.  Outputs CSV/JSON under results/analysis/luh/ (subsets) and
results/analysis/exp1/subset/ (tables).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import auprc, auroc, prr  # noqa: E402

from scripts.analysis.exp1_common import (  # noqa: E402
    DATASETS,
    METHODS,
    MODELS,
    RESULTS,
    answer_format,
    bootstrap_reps,
    fmt_ci,
    load_cell,
    metric_ci,
    write_csv,
)
from scripts.extract_per_model_subset import extract_per_model  # noqa: E402

SUBSET_DIR = RESULTS / "analysis" / "luh"
OUT = RESULTS / "analysis" / "exp1" / "subset"
TARGET = 200


def load_records_by_model(model: str) -> dict[str, dict]:
    """Joined records keyed by sample_id for one model (all datasets)."""
    out = {}
    for dataset in DATASETS:
        for rec in load_cell(model, dataset):
            out[rec["sample_id"]] = rec
    return out


def module_s1() -> dict[str, dict]:
    """Extract subsets and persist IDs (same format as the production script)."""
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    subsets = {}
    for model in MODELS:
        subsets[model] = extract_per_model(
            RESULTS / "uq", RESULTS / "judging", RESULTS / "generation", model, TARGET
        )
        ids = subsets[model]["positive_ids"] + subsets[model]["negative_ids"]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{model}: duplicate IDs in subset")
        (SUBSET_DIR / f"{model}_subset_ids.txt").write_text(
            "".join(f"{sid}\n" for sid in ids), encoding="utf-8"
        )
        print(
            f"{model}: positives={subsets[model]['n_positive']} "
            f"negatives={subsets[model]['n_negative']}"
        )
    (SUBSET_DIR / "per_model_subsets.json").write_text(
        json.dumps(subsets, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return subsets


def module_s2(subsets: dict[str, dict], records_by_model: dict[str, dict]) -> list[dict]:
    """Subset composition: dataset mix, correctness, rating, hallucination types."""
    rows = []
    for model in MODELS:
        sub = subsets[model]
        records = records_by_model[model]
        for name, ids, is_pos in (("luh_positive", sub["positive_ids"], True), ("matched_negative", sub["negative_ids"], False)):
            recs = [records[sid] for sid in ids]
            by_ds = {d: sum(1 for r in recs if r["dataset"] == d) for d in DATASETS}
            types = {"vision_hallucination": 0, "reasoning_hallucination": 0, "both": 0, "none": 0}
            for r in recs:
                t = set(r["hallucination_types"] or [])
                if len(t) >= 2:
                    types["both"] += 1
                elif t:
                    types[next(iter(t))] += 1
                else:
                    types["none"] += 1
            rows.append({
                "model": model, "group": name, "n": len(recs),
                **{f"n_{d}": by_ds[d] for d in DATASETS},
                "correct_share": sum(1 for r in recs if r["correct"]) / len(recs),
                "rating_mean": float(np.mean([r["rating"] or 0 for r in recs])),
                "rating_le2_share": sum(1 for r in recs if (r["rating"] or 0) <= 2) / len(recs),
                **{f"type_{k}": v for k, v in types.items()},
                "fmt_yesno_pct": sum(1 for r in recs if answer_format(r["answer"]) == "yes/no") / len(recs),
                "fmt_numeric_pct": sum(1 for r in recs if answer_format(r["answer"]) == "numeric") / len(recs),
                "token_count_mean": float(np.mean([r["token_count"] or 0 for r in recs])),
            })
    write_csv(OUT / "s2_subset_composition.csv", list(rows[0].keys()), rows)
    return rows


def module_s3(subsets: dict[str, dict], records_by_model: dict[str, dict]) -> list[dict]:
    """Matching quality: per-method pooled percentile means per group + distances."""
    rows = []
    for model in MODELS:
        sub = subsets[model]
        records = records_by_model[model]
        # recompute pooled percentiles on the same pool as the extraction script
        from scripts.extract_per_model_subset import _avg_rank_pct

        pool = [
            r for r in records.values()
            if r["has_uq"]
            and all(r["scores"].get(x) is not None for x in METHODS)
            and r["hallucination"] is not None
            and r["has_image"]
        ]
        sorted_vals = {m: sorted(r["scores"][m] for r in pool) for m in METHODS}
        for name, ids in (("luh_positive", sub["positive_ids"]), ("matched_negative", sub["negative_ids"])):
            row = {"model": model, "group": name, "n": len(ids)}
            for m in METHODS:
                pcts = [_avg_rank_pct(records[sid]["scores"][m], sorted_vals[m]) for sid in ids]
                row[f"pct_mean_{m}"] = float(np.mean(pcts))
                row[f"score_mean_{m}"] = float(np.mean([records[sid]["scores"][m] for sid in ids]))
            rows.append(row)
        # distances between matched pairs are recomputed from stored order
        pos_ids, neg_ids = sub["positive_ids"], sub["negative_ids"]
        dists = []
        for p, q in zip(pos_ids, neg_ids):
            d = np.sqrt(sum(
                (_avg_rank_pct(records[p]["scores"][m], sorted_vals[m]) -
                 _avg_rank_pct(records[q]["scores"][m], sorted_vals[m])) ** 2
                for m in METHODS
            ))
            dists.append(d)
        rows.append({
            "model": model, "group": "match_quality", "n": len(dists),
            "pair_dist_mean": float(np.mean(dists)),
            "pair_dist_median": float(np.median(dists)),
            "pair_dist_p90": float(np.percentile(dists, 90)),
        })
    header = ["model", "group", "n"] + [f"pct_mean_{m}" for m in METHODS] + \
             [f"score_mean_{m}" for m in METHODS] + \
             ["pair_dist_mean", "pair_dist_median", "pair_dist_p90"]
    write_csv(OUT / "s3_matching_quality.csv", header, rows)
    return rows


def module_s4(subsets: dict[str, dict], records_by_model: dict[str, dict]) -> list[dict]:
    """Baseline detection performance on the difficult subset (positive = LUH)."""
    rows = []
    metric_fns = {"auroc": auroc, "auprc": auprc, "prr": prr}
    for model in MODELS:
        sub = subsets[model]
        records = records_by_model[model]
        pos_ids = list(sub["positive_ids"])
        neg_ids = list(sub["negative_ids"])
        recs = [records[sid] for sid in pos_ids + neg_ids]
        labels = np.array([1] * len(pos_ids) + [0] * len(neg_ids), dtype=int)
        # sanity: positives must be hallucinated, negatives not
        if any(r["hallucination"] != bool(l) for r, l in zip(recs, labels)):
            raise ValueError(f"{model}: subset labels disagree with judge labels")
        reps = bootstrap_reps(recs)
        for method in METHODS:
            scores = np.array([r["scores"][method] for r in recs], dtype=float)
            row = {"model": model, "method": method, "n": len(recs), "n_pos": len(pos_ids)}
            for metric, fn in metric_fns.items():
                ci = metric_ci(fn, scores, labels, reps)
                row[metric] = ci["value"]
                row[f"{metric}_ci_low"] = ci["ci_low"]
                row[f"{metric}_ci_high"] = ci["ci_high"]
            rows.append(row)
    write_csv(OUT / "s4_subset_baseline.csv", list(rows[0].keys()), rows)
    return rows


def write_report(s2_rows, s4_rows) -> None:
    lines = ["# 3.5 困难子集汇总", ""]
    lines.append("## 组成（s2_subset_composition.csv）")
    lines.append("| 模型 | 组 | n | ViLP | HB | MM-Vet | 正确率 | rating≤2 | vision 型 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in s2_rows:
        lines.append(
            f"| {r['model']} | {r['group']} | {r['n']} | {r['n_vilp']} | "
            f"{r['n_hallusionbench']} | {r['n_mmvet']} | {r['correct_share']:.3f} | "
            f"{r['rating_le2_share']:.3f} | {r['type_vision_hallucination']} |"
        )
    lines += ["", "## 基线在子集上的 AUROC（s4_subset_baseline.csv）"]
    lines.append("| 模型 | PPL | SE | UMPIRE |")
    lines.append("|---|---|---|---|")
    by_model: dict[str, dict] = {}
    for r in s4_rows:
        by_model.setdefault(r["model"], {})[r["method"]] = r
    for model in MODELS:
        cells = [fmt_ci(by_model[model][m]["auroc"], by_model[model][m]) for m in METHODS]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    subsets = module_s1()
    records_by_model = {m: load_records_by_model(m) for m in MODELS}
    s2_rows = module_s2(subsets, records_by_model)
    s3_rows = module_s3(subsets, records_by_model)
    s4_rows = module_s4(subsets, records_by_model)
    write_report(s2_rows, s4_rows)
    print(f"subset analysis done -> {OUT}; subsets -> {SUBSET_DIR}")


if __name__ == "__main__":
    main()
