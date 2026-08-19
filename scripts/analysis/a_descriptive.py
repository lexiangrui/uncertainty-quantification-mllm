#!/usr/bin/env python3
"""Module A: descriptive statistics of the experiment-one results.

Outputs CSV tables under results/analysis/descriptive/ plus report.md.
No plotting: figure suggestions are documented in docs/实验一结果分析.md.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.load_joined import (  # noqa: E402
    DATASETS,
    METHODS,
    MODELS,
    RESULTS,
    cluster_ci,
    evaluated,
    load_cell,
    load_greedy,
    normalize_answer,
    write_csv,
)

OUT = RESULTS / "analysis" / "descriptive"


def _pct(n: int, total: int) -> float:
    return 100.0 * n / total if total else float("nan")


def module_a1() -> list[dict]:
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            greedy = load_greedy(model, dataset)
            records = load_cell(model, dataset)
            ev = evaluated(records)
            row = {
                "model": model,
                "dataset": dataset,
                "n_greedy": len(greedy),
                "n_sections_valid": sum(1 for g in greedy.values() if g["sections_valid"]),
                "n_sections_invalid": sum(1 for g in greedy.values() if not g["sections_valid"]),
                "n_judge_valid": len(records),
                "n_evaluated": len(ev),
            }
            for method in METHODS:
                row[f"n_{method}_valid"] = sum(1 for r in ev if r["scores"].get(method) is not None)
            row["n_all3_valid"] = sum(
                1 for r in ev if all(r["scores"].get(m) is not None for m in METHODS)
            )
            rows.append(row)
    write_csv(
        OUT / "a1_validity_flow.csv",
        list(rows[0].keys()),
        rows,
    )
    return rows


def module_a2() -> tuple[list[dict], list[dict]]:
    stats_rows, cross_rows = [], []
    for model in MODELS:
        for dataset in DATASETS:
            ev = evaluated(load_cell(model, dataset))
            n = len(ev)
            if not ev:
                stats_rows.append({
                    "model": model, "dataset": dataset, "n_greedy": len(greedy),
                    "n_sections_valid": sum(1 for g in greedy.values() if g["sections_valid"]),
                    "n_sections_invalid": sum(1 for g in greedy.values() if not g["sections_valid"]),
                    "n_judge_valid": len(records), "n_evaluated": 0,
                    **{f"n_{method}_valid": 0 for method in METHODS},
                    "n_all3_valid": 0,
                })
                continue
            correct = np.array([1 if r["correct"] else 0 for r in ev])
            hallu = np.array([1 if r["hallucination"] else 0 for r in ev])
            acc_ci = cluster_ci(lambda idx: float(correct[idx].mean()), ev)
            hr_ci = cluster_ci(lambda idx: float(hallu[idx].mean()), ev)
            stats_rows.append({
                "model": model,
                "dataset": dataset,
                "n_evaluated": n,
                "accuracy": float(correct.mean()),
                "acc_ci_low": acc_ci["ci_low"],
                "acc_ci_high": acc_ci["ci_high"],
                "hallu_rate": float(hallu.mean()),
                "hr_ci_low": hr_ci["ci_low"],
                "hr_ci_high": hr_ci["ci_high"],
            })
            # crosscheck against results/metrics labels
            metrics = json.loads((RESULTS / "metrics" / model / f"{dataset}.json").read_text())
            labels = metrics["labels"]
            for metric, value in (("accuracy", float(correct.mean())), ("hallucination_rate", float(hallu.mean()))):
                ref = labels[metric]["value"]
                cross_rows.append({
                    "model": model, "dataset": dataset, "metric": metric,
                    "value_analysis": value, "value_metrics": ref,
                    "abs_diff": abs(value - ref),
                })
    write_csv(
        OUT / "a2_label_stats.csv",
        list(stats_rows[0].keys()),
        stats_rows,
    )
    write_csv(
        OUT / "a2_crosscheck.csv",
        ["model", "dataset", "metric", "value_analysis", "value_metrics", "abs_diff"],
        cross_rows,
    )

    joint_rows, rating_rows, type_rows = [], [], []
    for model in MODELS:
        for dataset in DATASETS:
            ev = evaluated(load_cell(model, dataset))
            n = len(ev)
            for c in (0, 1):
                for h in (0, 1):
                    cnt = sum(1 for r in ev if int(bool(r["correct"])) == c and int(bool(r["hallucination"])) == h)
                    joint_rows.append({
                        "model": model, "dataset": dataset,
                        "correct": c, "hallucination": h,
                        "n": cnt, "pct": _pct(cnt, n),
                    })
            for rating in range(7):
                cnt = sum(1 for r in ev if r["rating"] == rating)
                rating_rows.append({
                    "model": model, "dataset": dataset, "rating": rating,
                    "n": cnt, "pct": _pct(cnt, n),
                })
            h1 = [r for r in ev if r["hallucination"]]
            def type_of(rec) -> str:
                types = set(rec["hallucination_types"] or [])
                if types == {"vision_hallucination"}:
                    return "vision_only"
                if types == {"reasoning_hallucination"}:
                    return "reasoning_only"
                if types >= {"vision_hallucination", "reasoning_hallucination"}:
                    return "both"
                return "unlabeled"
            for t in ("vision_only", "reasoning_only", "both", "unlabeled"):
                cnt = sum(1 for r in h1 if type_of(r) == t)
                type_rows.append({
                    "model": model, "dataset": dataset, "type": t,
                    "n": cnt, "pct": _pct(cnt, len(h1)),
                })
    write_csv(OUT / "a2_c_h_joint.csv", ["model", "dataset", "correct", "hallucination", "n", "pct"], joint_rows)
    write_csv(OUT / "a2_rating_dist.csv", ["model", "dataset", "rating", "n", "pct"], rating_rows)
    write_csv(OUT / "a2_hallu_types.csv", ["model", "dataset", "type", "n", "pct"], type_rows)
    return stats_rows, cross_rows


def module_a3() -> list[dict]:
    stat_rows, corr_rows = [], []
    pairs = [("perplexity", "semantic_entropy"), ("perplexity", "umpire"), ("semantic_entropy", "umpire")]
    for model in MODELS:
        for dataset in DATASETS:
            ev = evaluated(load_cell(model, dataset))
            for method in METHODS:
                vals = np.array([r["scores"][method] for r in ev if r["scores"].get(method) is not None])
                if len(vals) == 0:
                    continue
                q1, med, q3 = np.percentile(vals, [25, 50, 75])
                p5, p95 = np.percentile(vals, [5, 95])
                stat_rows.append({
                    "model": model, "dataset": dataset, "method": method, "n": len(vals),
                    "mean": float(vals.mean()), "std": float(vals.std()),
                    "median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
                    "skew": float(sps.skew(vals)), "p5": float(p5), "p95": float(p95),
                })
            common = [r for r in ev if all(r["scores"].get(m) is not None for m in METHODS)]
            for ma, mb in pairs:
                a = np.array([r["scores"][ma] for r in common])
                b = np.array([r["scores"][mb] for r in common])
                sp = sps.spearmanr(a, b)
                kt = sps.kendalltau(a, b)
                corr_rows.append({
                    "model": model, "dataset": dataset, "pair": f"{ma}|{mb}",
                    "spearman": float(sp.statistic), "spearman_p": float(sp.pvalue),
                    "kendall": float(kt.statistic), "kendall_p": float(kt.pvalue),
                    "n": len(common),
                })
    write_csv(
        OUT / "a3_score_stats.csv",
        ["model", "dataset", "method", "n", "mean", "std", "median", "q1", "q3", "iqr", "skew", "p5", "p95"],
        stat_rows,
    )
    write_csv(
        OUT / "a3_rank_corr.csv",
        ["model", "dataset", "pair", "spearman", "spearman_p", "kendall", "kendall_p", "n"],
        corr_rows,
    )
    return stat_rows


def module_a4() -> list[dict]:
    cluster_rows, majority_rows = [], []
    for model in MODELS:
        for dataset in DATASETS:
            ev = evaluated(load_cell(model, dataset, with_sample_answers=True))
            with_se = [r for r in ev if r["se_clusters"]]
            if with_se:
                n_clusters = [r["se_n_clusters"] for r in with_se]
                dominant = [r["se_dominant_mass"] for r in with_se]
                cluster_rows.append({
                    "model": model, "dataset": dataset, "n": len(with_se),
                    "mean_n_clusters": statistics.mean(n_clusters),
                    "median_n_clusters": statistics.median(n_clusters),
                    "pct_single_cluster": _pct(sum(1 for c in n_clusters if c == 1), len(with_se)),
                    "mean_dominant_mass": statistics.mean(dominant),
                    "median_dominant_mass": statistics.median(dominant),
                })
            for r in ev:
                r["_majority_match"] = majority_match(r)

            def group_records(key):
                if key == "all":
                    return ev
                if key == "correct":
                    return [r for r in ev if r["correct"]]
                if key == "wrong":
                    return [r for r in ev if not r["correct"]]
                if key == "hallu":
                    return [r for r in ev if r["hallucination"]]
                return [r for r in ev if not r["hallucination"]]

            for key in ("all", "correct", "wrong", "hallu", "no_hallu"):
                sub = group_records(key)
                if not sub:
                    continue
                arr = np.array([1 if r["_majority_match"] else 0 for r in sub])
                ci = cluster_ci(lambda idx: float(arr[idx].mean()), sub)
                majority_rows.append({
                    "model": model, "dataset": dataset, "group": key,
                    "n": len(sub), "match_rate": float(arr.mean()),
                    "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                })
    write_csv(
        OUT / "a4_se_cluster_stats.csv",
        ["model", "dataset", "n", "mean_n_clusters", "median_n_clusters", "pct_single_cluster",
         "mean_dominant_mass", "median_dominant_mass"],
        cluster_rows,
    )
    write_csv(
        OUT / "a4_greedy_majority.csv",
        ["model", "dataset", "group", "n", "match_rate", "ci_low", "ci_high"],
        majority_rows,
    )
    return cluster_rows


def majority_match(record: dict) -> bool:
    """Exact-normalized match of the greedy answer against majority-cluster members."""
    clusters = record.get("se_clusters") or []
    answers = record.get("sample_answers") or []
    if not clusters or not answers:
        return False
    majority = max(clusters, key=lambda c: c.get("probability", 0.0))
    target = normalize_answer(record.get("answer"))
    if not target:
        return False
    for member in majority.get("members", []):
        if isinstance(member, int) and 0 <= member < len(answers):
            if normalize_answer(answers[member]) == target:
                return True
    return False


def write_report(a1: list[dict], a2: list[dict], cross: list[dict], a3: list[dict]) -> None:
    max_diff = max(c["abs_diff"] for c in cross)
    n_eval_total = sum(r["n_evaluated"] for r in a2)
    acc_min = min(r["accuracy"] for r in a2)
    acc_max = max(r["accuracy"] for r in a2)
    hr_min = min(r["hallu_rate"] for r in a2)
    hr_max = max(r["hallu_rate"] for r in a2)
    import csv as _csv

    with open(OUT / "a3_rank_corr.csv", encoding="utf-8") as handle:
        corr_rows = list(_csv.DictReader(handle))
    spear_vals = [float(r["spearman"]) for r in corr_rows]
    lines = [
        "# 模块 A：描述性统计报告",
        "",
        "## A1 有效性",
        f"- 9 个单元格共 evaluated 样本 {n_eval_total} 条（Judge 有效且存在 UQ 记录）。",
        f"- 三方法全有效样本合计 {sum(r['n_all3_valid'] for r in a1)} 条。",
        "",
        "## A2 标签",
        f"- Accuracy 范围 {acc_min:.3f}–{acc_max:.3f}；Hallucination Rate 范围 {hr_min:.3f}–{hr_max:.3f}（9 格）。",
        f"- 与 results/metrics 的 accuracy/hallucination_rate 交叉核对最大绝对差 {max_diff:.2e}。",
        "",
        "## A3 分数分布",
        f"- 方法间 Spearman 秩相关范围 {min(spear_vals):.3f}–{max(spear_vals):.3f}（详见 a3_rank_corr.csv）。",
        "",
        "## A4 采样一致性",
        "- 详见 a4_se_cluster_stats.csv 与 a4_greedy_majority.csv。",
        "",
        "图表建议见 docs/实验一结果分析.md 各模块'可绘图'条目。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a1 = module_a1()
    a2, cross = module_a2()
    a3 = module_a3()
    a4 = module_a4()
    write_report(a1, a2, cross, a3)
    print(f"module A done -> {OUT}")
    bad = [c for c in cross if c["abs_diff"] > 1e-9]
    print(f"crosscheck vs results/metrics: {len(cross) - len(bad)}/{len(cross)} consistent, max_diff={max(c['abs_diff'] for c in cross):.2e}")
    if bad:
        for c in bad:
            print("MISMATCH:", c)
        sys.exit(1)


if __name__ == "__main__":
    main()
