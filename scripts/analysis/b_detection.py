#!/usr/bin/env python3
"""Module B: detection-performance comparison of the three UQ methods.

Targets: error (E = 1 - correct) and hallucination (H). Metrics reuse
src/evaluation/metrics.py so point estimates can be crosschecked against
results/metrics exactly. Outputs CSV tables under results/analysis/detection/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import auroc, auprc, ece, prr  # noqa: E402
from scripts.analysis.load_joined import (  # noqa: E402
    DATASETS,
    METHODS,
    MODELS,
    RESULTS,
    all3_valid,
    cluster_ci,
    evaluated,
    load_cell,
    write_csv,
)

OUT = RESULTS / "analysis" / "detection"

N_BOOT = 1000
SEED = 0
ECE_BINS = 15
METRICS = ("auroc", "auprc", "prr", "ece")
METRIC_FNS = {"auroc": auroc, "auprc": auprc, "prr": prr, "ece": lambda s, l: ece(s, l, bins=ECE_BINS)}
TARGETS = ("error", "hallucination")


def labels_for(target: str, records: list[dict]) -> np.ndarray:
    if target == "error":
        return np.array([0 if r["correct"] else 1 for r in records], dtype=int)
    return np.array([1 if r["hallucination"] else 0 for r in records], dtype=int)


def replicate_values(fn, scores: np.ndarray, labels: np.ndarray, reps: list[np.ndarray]) -> list:
    return [fn(scores[idx], labels[idx]) for idx in reps]


def percentile_ci(values: list) -> dict:
    defined = [v for v in values if v is not None]
    undefined = len(values) - len(defined)
    if not defined:
        return {"ci_low": None, "ci_high": None, "undefined_replicates": undefined}
    low, high = np.percentile(defined, (2.5, 97.5))
    return {"ci_low": float(low), "ci_high": float(high), "undefined_replicates": undefined}


def avg_rank_pct(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg / len(values)
        i = j + 1
    return ranks


def load_cells() -> dict[tuple[str, str], list[dict]]:
    cells = {}
    for model in MODELS:
        for dataset in DATASETS:
            cells[(model, dataset)] = evaluated(load_cell(model, dataset))
    return cells


def module_b1(cells: dict[tuple[str, str], list[dict]]) -> tuple[list[dict], list[dict]]:
    rows, cross = [], []
    for (model, dataset), records in cells.items():
        metrics_json = json.loads((RESULTS / "metrics" / model / f"{dataset}.json").read_text())
        for target in TARGETS:
            labels_all = labels_for(target, records)
            for method in METHODS:
                idx_valid = [i for i, r in enumerate(records) if r["scores"].get(method) is not None]
                sub = [records[i] for i in idx_valid]
                scores = np.array([records[i]["scores"][method] for i in idx_valid], dtype=float)
                labels = labels_all[idx_valid]
                groups = [r["group_id"] for r in sub]
                from src.evaluation.metrics import cluster_bootstrap_indices

                reps = cluster_bootstrap_indices(groups, n_bootstrap=N_BOOT, seed=SEED)
                row = {
                    "model": model, "dataset": dataset, "target": target, "method": method,
                    "n": len(sub), "n_pos": int(labels.sum()),
                    "pos_rate": float(labels.mean()) if len(sub) else None,
                }
                for metric in METRICS:
                    fn = METRIC_FNS[metric]
                    point = fn(scores, labels)
                    ci = percentile_ci(replicate_values(fn, scores, labels, reps))
                    row[metric] = point
                    row[f"{metric}_ci_low"] = ci["ci_low"]
                    row[f"{metric}_ci_high"] = ci["ci_high"]
                rows.append(row)
                # crosscheck point estimates against results/metrics
                ref = metrics_json["targets"][target]["methods"][method]
                for metric in METRICS:
                    cross.append({
                        "model": model, "dataset": dataset, "target": target, "method": method,
                        "metric": metric,
                        "value_analysis": row[metric],
                        "value_metrics": ref[metric]["value"],
                        "abs_diff": abs(row[metric] - ref[metric]["value"])
                        if row[metric] is not None and ref[metric]["value"] is not None
                        else None,
                    })
    header = list(rows[0].keys())
    write_csv(OUT / "b1_metrics_main.csv", header, rows)
    write_csv(
        OUT / "b1_crosscheck.csv",
        ["model", "dataset", "target", "method", "metric", "value_analysis", "value_metrics", "abs_diff"],
        cross,
    )
    return rows, cross


def module_b2(cells: dict[tuple[str, str], list[dict]], b1_rows: list[dict]) -> list[dict]:
    from src.evaluation.metrics import cluster_bootstrap_indices

    pairs = [("semantic_entropy", "perplexity"), ("umpire", "perplexity"), ("umpire", "semantic_entropy")]
    diff_rows, gap_rows = [], []
    cell_arrays: dict[tuple, dict] = {}
    for (model, dataset), records in cells.items():
        common = all3_valid(records)
        groups = [r["group_id"] for r in common]
        reps = cluster_bootstrap_indices(groups, n_bootstrap=N_BOOT, seed=SEED)
        arrays = {}
        for target in TARGETS:
            labels = labels_for(target, common)
            for method in METHODS:
                scores = np.array([r["scores"][method] for r in common], dtype=float)
                arrays[(target, method)] = (scores, labels)
        cell_arrays[(model, dataset)] = (common, reps, arrays)

        for target in TARGETS:
            for ma, mb in pairs:
                sa, la = arrays[(target, ma)]
                sb, lb = arrays[(target, mb)]
                for metric in METRICS:
                    fn = METRIC_FNS[metric]
                    point = fn(sa, la) - fn(sb, lb) if fn(sa, la) is not None and fn(sb, lb) is not None else None
                    deltas = []
                    for idx in reps:
                        va = fn(sa[idx], la[idx])
                        vb = fn(sb[idx], lb[idx])
                        deltas.append(None if va is None or vb is None else va - vb)
                    ci = percentile_ci(deltas)
                    diff_rows.append({
                        "model": model, "dataset": dataset, "target": target,
                        "pair": f"{ma}-{mb}", "metric": metric,
                        "delta": point, "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                    })
        # E vs H target gap per method (AUROC only)
        for method in METHODS:
            se, le = arrays[("error", method)]
            sh, lh = arrays[("hallucination", method)]
            ae, ah = auroc(se, le), auroc(sh, lh)
            gaps = []
            for idx in reps:
                ve, vh = auroc(se[idx], le[idx]), auroc(sh[idx], lh[idx])
                gaps.append(None if ve is None or vh is None else vh - ve)
            ci = percentile_ci(gaps)
            gap_rows.append({
                "model": model, "dataset": dataset, "method": method,
                "auroc_error": ae, "auroc_hallucination": ah,
                "gap_h_minus_e": None if ae is None or ah is None else ah - ae,
                "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            })
    write_csv(
        OUT / "b2_method_diff.csv",
        ["model", "dataset", "target", "pair", "metric", "delta", "ci_low", "ci_high"],
        diff_rows,
    )
    write_csv(
        OUT / "b2_target_gap.csv",
        ["model", "dataset", "method", "auroc_error", "auroc_hallucination", "gap_h_minus_e", "ci_low", "ci_high"],
        gap_rows,
    )

    # rank stability on AUROC point estimates from b1
    rank_rows = []
    for target in TARGETS:
        wins = {m: 0 for m in METHODS}
        ranks = {m: [] for m in METHODS}
        for model in MODELS:
            for dataset in DATASETS:
                cell = [r for r in b1_rows if r["target"] == target and r["model"] == model and r["dataset"] == dataset]
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
    write_csv(OUT / "b2_rank_stability.csv", ["target", "method", "n_cells", "n_best_auroc", "mean_rank"], rank_rows)

    # macro-average AUROC per model across datasets (replicate-index-paired mean)
    macro_rows = []
    for model in MODELS:
        for target in TARGETS:
            for method in METHODS:
                per_dataset, points = [], []
                for dataset in DATASETS:
                    common, reps, arrays = cell_arrays[(model, dataset)]
                    scores, labels = arrays[(target, method)]
                    vals = replicate_values(auroc, scores, labels, reps)
                    per_dataset.append(vals)
                    points.append(auroc(scores, labels))
                macro = [
                    np.mean([d[i] for d in per_dataset if d[i] is not None])
                    if any(d[i] is not None for d in per_dataset) else None
                    for i in range(N_BOOT)
                ]
                ci = percentile_ci(macro)
                macro_rows.append({
                    "model": model, "target": target, "method": method,
                    "macro_auroc": float(np.mean(points)),
                    "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                })
    write_csv(OUT / "b2_macro.csv", ["model", "target", "method", "macro_auroc", "ci_low", "ci_high"], macro_rows)
    return diff_rows, gap_rows


def module_b3(cells: dict[tuple[str, str], list[dict]]) -> None:
    bin_rows, rc_rows = [], []
    grid = np.round(np.arange(0, 1.0, 0.01), 4)
    for (model, dataset), records in cells.items():
        for target in TARGETS:
            labels_all = labels_for(target, records)
            for method in METHODS:
                idx_valid = [i for i, r in enumerate(records) if r["scores"].get(method) is not None]
                scores = np.array([records[i]["scores"][method] for i in idx_valid], dtype=float)
                labels = labels_all[idx_valid]
                n = len(scores)
                # ECE bins on min-max normalised scores
                lo, hi = float(scores.min()), float(scores.max())
                norm = np.zeros(n) if hi == lo else (scores - lo) / (hi - lo)
                for b in range(ECE_BINS):
                    edge_lo, edge_hi = b / ECE_BINS, (b + 1) / ECE_BINS
                    mask = (norm >= edge_lo) & (norm < edge_hi) if b < ECE_BINS - 1 else (norm >= edge_lo) & (norm <= 1.0)
                    cnt = int(mask.sum())
                    if cnt == 0:
                        bin_rows.append({
                            "model": model, "dataset": dataset, "target": target, "method": method,
                            "bin": b, "bin_lo": edge_lo, "bin_hi": edge_hi, "n": 0,
                            "mean_score_norm": None, "actual_rate": None, "abs_gap": None, "weighted_contrib": 0.0,
                        })
                        continue
                    mean_pred = float(norm[mask].mean())
                    actual = float(labels[mask].mean())
                    gap = abs(mean_pred - actual)
                    bin_rows.append({
                        "model": model, "dataset": dataset, "target": target, "method": method,
                        "bin": b, "bin_lo": edge_lo, "bin_hi": edge_hi, "n": cnt,
                        "mean_score_norm": mean_pred, "actual_rate": actual,
                        "abs_gap": gap, "weighted_contrib": cnt / n * gap,
                    })
                # risk-coverage curve on descending score order
                order = np.argsort(-scores, kind="mergesort")
                sorted_labels = labels[order]
                neg_cum = np.cumsum(1 - sorted_labels)
                for frac in grid:
                    k = int(round(frac * n))
                    if k >= n:
                        continue
                    retained_n = n - k
                    rc_rows.append({
                        "model": model, "dataset": dataset, "target": target, "method": method,
                        "rejected_frac": float(frac), "rejected_k": k, "retained_n": retained_n,
                        "retained_precision": float(neg_cum[retained_n - 1] / retained_n),
                    })
    write_csv(
        OUT / "b3_ece_bins.csv",
        ["model", "dataset", "target", "method", "bin", "bin_lo", "bin_hi", "n",
         "mean_score_norm", "actual_rate", "abs_gap", "weighted_contrib"],
        bin_rows,
    )
    write_csv(
        OUT / "b3_risk_coverage.csv",
        ["model", "dataset", "target", "method", "rejected_frac", "rejected_k", "retained_n", "retained_precision"],
        rc_rows,
    )


def module_b4(cells: dict[tuple[str, str], list[dict]]) -> None:
    group_rows, pct_rows, sub_rows = [], [], []
    group_defs = [("c1_h0", 1, 0), ("c1_h1", 1, 1), ("c0_h0", 0, 0), ("c0_h1", 0, 1)]
    for (model, dataset), records in cells.items():
        common = all3_valid(records)
        for method in METHODS:
            scores = np.array([r["scores"][method] for r in common], dtype=float)
            pcts = avg_rank_pct(scores)
            for name, c, h in group_defs:
                idx = [i for i, r in enumerate(common) if int(bool(r["correct"])) == c and int(bool(r["hallucination"])) == h]
                if not idx:
                    continue
                vals = scores[idx]
                group_rows.append({
                    "model": model, "dataset": dataset, "method": method, "group": name, "n": len(idx),
                    "mean": float(vals.mean()), "p25": float(np.percentile(vals, 25)),
                    "p50": float(np.percentile(vals, 50)), "p75": float(np.percentile(vals, 75)),
                })
                pct_rows.append({
                    "model": model, "dataset": dataset, "method": method, "group": name, "n": len(idx),
                    "mean_pct": float(pcts[idx].mean()),
                })
        # subgroup AUROC: positives = wrong samples split by H, negatives = correct samples
        neg_idx = [i for i, r in enumerate(common) if r["correct"]]
        for method in METHODS:
            scores = np.array([r["scores"][method] for r in common], dtype=float)
            for comp, h in (("wrong_with_hallu_vs_correct", 1), ("wrong_no_hallu_vs_correct", 0)):
                pos_idx = [i for i, r in enumerate(common) if not r["correct"] and int(bool(r["hallucination"])) == h]
                if not pos_idx or not neg_idx:
                    continue
                sel = pos_idx + neg_idx
                s = scores[sel]
                l = np.array([1] * len(pos_idx) + [0] * len(neg_idx), dtype=int)
                sub_recs = [common[i] for i in sel]
                ci = cluster_ci(lambda idx: auroc(s[idx], l[idx]), sub_recs)
                sub_rows.append({
                    "model": model, "dataset": dataset, "method": method, "comparison": comp,
                    "n_pos": len(pos_idx), "n_neg": len(neg_idx),
                    "auroc": auroc(s, l), "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                })
    write_csv(
        OUT / "b4_group_scores.csv",
        ["model", "dataset", "method", "group", "n", "mean", "p25", "p50", "p75"],
        group_rows,
    )
    write_csv(OUT / "b4_group_pct.csv", ["model", "dataset", "method", "group", "n", "mean_pct"], pct_rows)
    write_csv(
        OUT / "b4_subgroup_auroc.csv",
        ["model", "dataset", "method", "comparison", "n_pos", "n_neg", "auroc", "ci_low", "ci_high"],
        sub_rows,
    )


def write_report(cross: list[dict], gap_rows: list[dict]) -> None:
    diffs = [c["abs_diff"] for c in cross if c["abs_diff"] is not None]
    lines = [
        "# 模块 B：检测性能对比报告",
        "",
        "## 与 results/metrics 的交叉核对",
        f"- {len(diffs)} 项点估计全部对比，最大绝对差 {max(diffs):.2e}（容差 1e-9）。",
        "- 说明：CI 区间因重采样簇序可能与 metrics 略有差异，核对以点估计为准。",
        "",
        "## 方法排名（按 AUROC，9 格）",
    ]
    import csv as _csv

    with open(OUT / "b2_rank_stability.csv", encoding="utf-8") as handle:
        for r in _csv.DictReader(handle):
            lines.append(
                f"- target={r['target']}, method={r['method']}: "
                f"最优 {r['n_best_auroc']}/9 格，平均排名 {float(r['mean_rank']):.2f}"
            )
    gaps = [g["gap_h_minus_e"] for g in gap_rows if g["gap_h_minus_e"] is not None]
    lines += [
        "",
        "## E vs H 目标差距（AUROC(H) − AUROC(E)，b2_target_gap.csv）",
        f"- 27 个 格×方法 的差距均值 {np.mean(gaps):.3f}，范围 [{min(gaps):.3f}, {max(gaps):.3f}]。",
        "",
        "图表建议见 docs/实验一结果分析.md 各模块'可绘图'条目。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    b1_rows, cross = module_b1(cells)
    _, gap_rows = module_b2(cells, b1_rows)
    module_b3(cells)
    module_b4(cells)
    write_report(cross, gap_rows)
    print(f"module B done -> {OUT}")
    bad = [c for c in cross if c["abs_diff"] is None or c["abs_diff"] > 1e-9]
    print(f"crosscheck vs results/metrics: {len(cross) - len(bad)}/{len(cross)} consistent")
    for c in bad[:10]:
        print("MISMATCH:", c)
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
