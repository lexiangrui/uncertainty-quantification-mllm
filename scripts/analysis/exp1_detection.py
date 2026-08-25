#!/usr/bin/env python3
"""Section 3.2: detection performance of the three UQ methods.

Targets:
- error (E = 1 - C): answer-error detection;
- hallucination (H): hallucination detection;
- hallucination_given_error: among wrong answers only, H vs non-H.

Metrics reuse src/evaluation.metrics so point estimates crosscheck
results/metrics exactly.  Outputs CSV tables under
results/analysis/exp1/detection/.
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
    bootstrap_reps,
    evaluated,
    fmt_ci,
    load_cell,
    metric_ci,
    write_csv,
)

OUT = RESULTS / "analysis" / "exp1" / "detection"

METRICS = ("auroc", "auprc", "prr")
METRIC_FNS = {"auroc": auroc, "auprc": auprc, "prr": prr}
TARGETS = ("error", "hallucination", "hallucination_given_error")


def labels_for(target: str, records: list[dict]) -> np.ndarray:
    if target == "error":
        return np.array([0 if r["correct"] else 1 for r in records], dtype=int)
    if target == "hallucination":
        return np.array([1 if r["hallucination"] else 0 for r in records], dtype=int)
    wrong = [r for r in records if not r["correct"]]
    return np.array([1 if r["hallucination"] else 0 for r in wrong], dtype=int)


def module_t1(cells) -> tuple[list[dict], list[dict]]:
    """Per-cell detection metrics with clustered bootstrap CIs + metrics crosscheck."""
    rows, cross = [], []
    for (model, dataset), records in cells.items():
        metrics_json = json.loads((RESULTS / "metrics" / model / f"{dataset}.json").read_text())
        for target in TARGETS:
            labels_all = labels_for(target, records)
            for method in METHODS:
                idx = [i for i, r in enumerate(records) if r["has_uq"] and r["scores"].get(method) is not None]
                if target == "hallucination_given_error":
                    idx = [i for i in idx if not records[i]["correct"]]
                sub = [records[i] for i in idx]
                if not sub:
                    continue
                scores = np.array([records[i]["scores"][method] for i in idx], dtype=float)
                if target == "hallucination_given_error":
                    labels = np.array([1 if records[i]["hallucination"] else 0 for i in idx], dtype=int)
                else:
                    labels = labels_all[idx]
                reps = bootstrap_reps(sub)
                row = {
                    "model": model, "dataset": dataset, "target": target, "method": method,
                    "n": len(sub), "n_pos": int(labels.sum()),
                    "pos_rate": float(labels.mean()),
                }
                for metric in METRICS:
                    ci = metric_ci(METRIC_FNS[metric], scores, labels, reps)
                    row[metric] = ci["value"]
                    row[f"{metric}_ci_low"] = ci["ci_low"]
                    row[f"{metric}_ci_high"] = ci["ci_high"]
                rows.append(row)
                if target in ("error", "hallucination"):
                    ref = metrics_json["targets"][target]["methods"][method]
                    for metric in METRICS:
                        cross.append({
                            "model": model, "dataset": dataset, "target": target, "method": method,
                            "metric": metric,
                            "value_analysis": row[metric],
                            "value_metrics": ref[metric]["value"],
                            "abs_diff": abs(row[metric] - ref[metric]["value"]),
                        })
    write_csv(OUT / "t1_detection_metrics.csv", list(rows[0].keys()), rows)
    write_csv(
        OUT / "t1_crosscheck.csv",
        ["model", "dataset", "target", "method", "metric", "value_analysis", "value_metrics", "abs_diff"],
        cross,
    )
    return rows, cross


def module_t2(cells, t1_rows) -> list[dict]:
    """E-vs-H target gap (paired clustered bootstrap) plus rank/macro summaries."""
    gap_rows = []
    for (model, dataset), records in cells.items():
        pool = [r for r in records if r["has_uq"]]
        reps = bootstrap_reps(pool)
        for method in METHODS:
            idx = [i for i, r in enumerate(pool) if r["scores"].get(method) is not None]
            scores = np.array([pool[i]["scores"][method] for i in idx], dtype=float)
            le = labels_for("error", pool)[idx]
            lh = labels_for("hallucination", pool)[idx]
            ae = auroc(scores, le) if len(np.unique(le)) == 2 else None
            ah = auroc(scores, lh) if len(np.unique(lh)) == 2 else None
            gaps = []
            for rep in reps:
                s = scores[rep]
                e, h = le[rep], lh[rep]
                ve = auroc(s, e) if len(np.unique(e)) == 2 else None
                vh = auroc(s, h) if len(np.unique(h)) == 2 else None
                gaps.append(None if ve is None or vh is None else vh - ve)
            ci = _pct_ci(gaps)
            gap_rows.append({
                "model": model, "dataset": dataset, "method": method,
                "auroc_error": ae, "auroc_hallucination": ah,
                "gap_h_minus_e": None if ae is None or ah is None else ah - ae,
                "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            })
    write_csv(
        OUT / "t2_target_gap.csv",
        ["model", "dataset", "method", "auroc_error", "auroc_hallucination", "gap_h_minus_e", "ci_low", "ci_high"],
        gap_rows,
    )

    # rank stability and macro averages on AUROC point estimates
    rank_rows = []
    for target in ("error", "hallucination"):
        wins = {m: 0 for m in METHODS}
        ranks = {m: [] for m in METHODS}
        for model in MODELS:
            for dataset in DATASETS:
                cell = [
                    r for r in t1_rows
                    if r["target"] == target and r["model"] == model and r["dataset"] == dataset
                ]
                order = sorted(cell, key=lambda r: -(r["auroc"] if r["auroc"] is not None else -1))
                for rank, r in enumerate(order, start=1):
                    ranks[r["method"]].append(rank)
                    if rank == 1:
                        wins[r["method"]] += 1
        for method in METHODS:
            rank_rows.append({
                "target": target, "method": method,
                "n_cells": len(ranks[method]),
                "n_best_auroc": wins[method],
                "mean_rank": float(np.mean(ranks[method])),
            })
    write_csv(OUT / "t2_rank_stability.csv", ["target", "method", "n_cells", "n_best_auroc", "mean_rank"], rank_rows)

    macro_rows = []
    for model in MODELS:
        for target in ("error", "hallucination"):
            for method in METHODS:
                pts, per_dataset = [], []
                for dataset in DATASETS:
                    rec = next(
                        r for r in t1_rows
                        if r["model"] == model and r["dataset"] == dataset
                        and r["target"] == target and r["method"] == method
                    )
                    pts.append(rec["auroc"])
                    per_dataset.append(rec)
                macro_rows.append({
                    "model": model, "target": target, "method": method,
                    "macro_auroc": float(np.mean(pts)),
                    "macro_auprc": float(np.mean([r["auprc"] for r in per_dataset])),
                    "macro_prr": float(np.mean([r["prr"] for r in per_dataset])),
                })
    write_csv(OUT / "t2_macro.csv", ["model", "target", "method", "macro_auroc", "macro_auprc", "macro_prr"], macro_rows)
    return gap_rows


def _pct_ci(values: list) -> dict:
    defined = [v for v in values if v is not None]
    if not defined:
        return {"ci_low": None, "ci_high": None}
    low, high = np.percentile(defined, (2.5, 97.5))
    return {"ci_low": float(low), "ci_high": float(high)}


def write_report(t1_rows, gap_rows, cross) -> None:
    lines = ["# 3.2 检测性能汇总", ""]
    lines.append("## AUROC 主表（t1_detection_metrics.csv，目标=error / hallucination）")
    lines.append("| 模型 × 数据集 | 目标 | PPL | SE | UMPIRE |")
    lines.append("|---|---|---|---|---|")
    import csv as _csv

    with open(OUT / "t1_detection_metrics.csv", encoding="utf-8") as handle:
        rows = list(_csv.DictReader(handle))
    by_key: dict[tuple, dict] = {}
    for r in rows:
        by_key[(r["model"], r["dataset"], r["target"], r["method"])] = r
    for model in MODELS:
        for dataset in DATASETS:
            for target in ("error", "hallucination"):
                cells = []
                for method in METHODS:
                    r = by_key.get((model, dataset, target, method))
                    cells.append(f"{float(r['auroc']):.3f}" if r else "N/A")
                lines.append(f"| {model} / {dataset} | {target} | " + " | ".join(cells) + " |")
    gaps = [g["gap_h_minus_e"] for g in gap_rows if g["gap_h_minus_e"] is not None]
    diffs = [c["abs_diff"] for c in cross if c["abs_diff"] is not None]
    lines += [
        "",
        f"## E vs H 目标差距：27 个单元格×方法的 AUROC(H)−AUROC(E) 均值 "
        f"{np.mean(gaps):.3f}，范围 [{min(gaps):.3f}, {max(gaps):.3f}]。",
        "",
        f"与 results/metrics 交叉核对：{len(diffs)} 项点估计，最大绝对差 {max(diffs):.2e}。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = {}
    for model in MODELS:
        for dataset in DATASETS:
            cells[(model, dataset)] = evaluated(load_cell(model, dataset))
    t1_rows, cross = module_t1(cells)
    gap_rows = module_t2(cells, t1_rows)
    write_report(t1_rows, gap_rows, cross)
    bad = [c for c in cross if c["abs_diff"] > 1e-9]
    print(f"detection done -> {OUT}")
    print(f"crosscheck vs results/metrics: {len(cross) - len(bad)}/{len(cross)} consistent")
    for c in bad:
        print("MISMATCH:", c)
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
