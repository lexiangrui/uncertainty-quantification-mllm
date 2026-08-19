#!/usr/bin/env python3
"""Module C1: scale quantification, miss attribution and profiling of
low-uncertainty hallucinations (LUH).

Groups per model on the extraction pool (judge-valid + all three UQ scores
valid + has_image, three datasets merged):
- luh:            200 positive ids from results/analysis/luh/
- matched_neg:    200 matched negative ids from the same file
- detected_hallu: pool samples with H=1 that are not in luh
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
    all3_valid,
    cluster_ci,
    evaluated,
    load_cell,
    normalize_answer,
    write_csv,
)

OUT = RESULTS / "analysis" / "luh_profile"
SUBSETS = RESULTS / "analysis" / "luh" / "per_model_subsets.json"


def load_pool() -> dict[str, list[dict]]:
    pool: dict[str, list[dict]] = {}
    subsets = json.loads(SUBSETS.read_text())
    for model in MODELS:
        records: list[dict] = []
        for dataset in DATASETS:
            records.extend(all3_valid(evaluated(load_cell(model, dataset))))
        records = [r for r in records if r["has_image"]]
        luh = set(subsets[model]["positive_ids"])
        neg = set(subsets[model]["negative_ids"])
        assert len(luh) == 200 and len(neg) == 200, f"{model}: subset size != 200"
        for rec in records:
            if rec["sample_id"] in luh:
                rec["group"] = "luh"
            elif rec["sample_id"] in neg:
                rec["group"] = "matched_neg"
            elif rec["hallucination"]:
                rec["group"] = "detected_hallu"
            else:
                rec["group"] = "other"
        found = sum(1 for r in records if r["group"] == "luh")
        found_neg = sum(1 for r in records if r["group"] == "matched_neg")
        assert found == 200 and found_neg == 200, (
            f"{model}: pool only covers {found}/{found_neg} of the 200/200 subset ids"
        )
        pool[model] = records
    return pool


def _avg_rank_pct(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = ((i + j) / 2.0 + 1.0) / len(values)
        i = j + 1
    return ranks


LUH_ALPHAS = (0.25, 0.50)


def _method_arrays(records: list[dict], method: str):
    sub = [r for r in records if r["scores"].get(method) is not None]
    scores = np.array([r["scores"][method] for r in sub], dtype=float)
    h = np.array([1 if r["hallucination"] else 0 for r in sub], dtype=int)
    return sub, scores, h


def luh_share(scores: np.ndarray, h: np.ndarray, alpha: float):
    """P(s_H1 <= Q_alpha(H0)): share of hallucinated samples whose uncertainty
    score is at or below the alpha quantile of the non-hallucination score
    distribution."""
    h0, h1 = scores[h == 0], scores[h == 1]
    if h0.size == 0 or h1.size == 0:
        return None, None
    threshold = float(np.quantile(h0, alpha))
    return float(np.mean(h1 <= threshold)), threshold


def module_c11(pooled: dict[str, list[dict]]) -> list[dict]:
    """LUH-scale shares (alpha = 0.25, 0.50) per model x method on the pooled
    evaluated set, with group-level bootstrap CIs."""
    rows = []
    for model in MODELS:
        for method in METHODS:
            sub, scores, h = _method_arrays(pooled[model], method)
            for alpha in LUH_ALPHAS:
                rate, threshold = luh_share(scores, h, alpha)
                ci = cluster_ci(lambda idx: luh_share(scores[idx], h[idx], alpha)[0], sub)
                rows.append({
                    "model": model, "method": method, "alpha": alpha,
                    "n_evaluated": len(sub), "n_h1": int(h.sum()), "n_h0": int((1 - h).sum()),
                    "threshold": threshold,
                    "luh_share": rate, "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                    "null_exchangeable": alpha,
                    "excess": None if rate is None else rate - alpha,
                })
    write_csv(
        OUT / "c1_luh_rate.csv",
        ["model", "method", "alpha", "n_evaluated", "n_h1", "n_h0", "threshold",
         "luh_share", "ci_low", "ci_high", "null_exchangeable", "excess"],
        rows,
    )
    return rows


def _tokens(rec: dict) -> int:
    if isinstance(rec.get("token_count"), int) and rec["token_count"] > 0:
        return rec["token_count"]
    return len((rec.get("answer") or "").split())


def _answer_class(rec: dict) -> str:
    norm = normalize_answer(rec.get("answer"))
    if norm in {"yes", "no"}:
        return "yesno"
    try:
        float(norm.replace(",", ""))
        return "numeric"
    except ValueError:
        pass
    if _tokens(rec) <= 3:
        return "short"
    return "long"


def module_c12(pool: dict[str, list[dict]]) -> None:
    fmt_rows, corr_rows, se_rows, ump_rows = [], [], [], []
    groups = ("luh", "matched_neg", "detected_hallu")
    for model in MODELS:
        by_group = {g: [r for r in pool[model] if r["group"] == g] for g in groups}
        for g in groups:
            recs = by_group[g]
            tokens = [_tokens(r) for r in recs]
            classes = [_answer_class(r) for r in recs]
            n = len(recs)
            fmt_rows.append({
                "model": model, "group": g, "n": n,
                "mean_tokens": statistics.mean(tokens),
                "median_tokens": statistics.median(tokens),
                "pct_yesno": 100.0 * classes.count("yesno") / n,
                "pct_numeric": 100.0 * classes.count("numeric") / n,
                "pct_short": 100.0 * classes.count("short") / n,
                "pct_long": 100.0 * classes.count("long") / n,
            })
            with_se = [r for r in recs if r["se_n_clusters"] is not None]
            if with_se:
                n_clusters = [r["se_n_clusters"] for r in with_se]
                dominant = [r["se_dominant_mass"] for r in with_se]
                se_rows.append({
                    "model": model, "group": g, "n": len(with_se),
                    "mean_n_clusters": statistics.mean(n_clusters),
                    "pct_single_cluster": 100.0 * sum(1 for c in n_clusters if c == 1) / len(with_se),
                    "mean_dominant_mass": statistics.mean(dominant),
                })
            with_ump = [r for r in recs if r["ump_semantic_volume"] is not None]
            if with_ump:
                ump_rows.append({
                    "model": model, "group": g, "n": len(with_ump),
                    "mean_semantic_volume": statistics.mean(r["ump_semantic_volume"] for r in with_ump),
                    "mean_incoherence_mean": statistics.mean(r["ump_incoherence_mean"] for r in with_ump),
                })
        # PPL vs answer length rank correlation within H=1 groups
        for g in groups + ("all_h1",):
            recs = by_group[g] if g != "all_h1" else [r for r in pool[model] if r["hallucination"]]
            if len(recs) < 3:
                continue
            ppl = [r["scores"]["perplexity"] for r in recs]
            tok = [_tokens(r) for r in recs]
            corr = sps.spearmanr(ppl, tok)
            corr_rows.append({
                "model": model, "subset": g, "n": len(recs),
                "spearman": float(corr.statistic), "p_value": float(corr.pvalue),
            })
    write_csv(
        OUT / "c1_answer_format.csv",
        ["model", "group", "n", "mean_tokens", "median_tokens",
         "pct_yesno", "pct_numeric", "pct_short", "pct_long"],
        fmt_rows,
    )
    write_csv(OUT / "c1_ppl_length_corr.csv", ["model", "subset", "n", "spearman", "p_value"], corr_rows)
    write_csv(
        OUT / "c1_se_clusters.csv",
        ["model", "group", "n", "mean_n_clusters", "pct_single_cluster", "mean_dominant_mass"],
        se_rows,
    )
    write_csv(
        OUT / "c1_umpire_components.csv",
        ["model", "group", "n", "mean_semantic_volume", "mean_incoherence_mean"],
        ump_rows,
    )


def _profile_rows(pool: dict[str, list[dict]], dim_fn, values, *, h1_only=False) -> list[dict]:
    rows = []
    for model in MODELS:
        for group in ("luh", "matched_neg", "detected_hallu"):
            recs = [r for r in pool[model] if r["group"] == group]
            if h1_only:
                recs = [r for r in recs if r["hallucination"]]
            if not recs:
                continue
            for value in values:
                arr = np.array([1 if dim_fn(r) == value else 0 for r in recs])
                if arr.sum() == 0 and value != values[0] and value != "__all__":
                    continue
                if arr.sum() == 0:
                    continue
                ci = cluster_ci(lambda idx: float(arr[idx].mean()), recs)
                rows.append({
                    "model": model, "group": group, "value": str(value),
                    "n": int(arr.sum()), "pct": float(arr.mean() * 100),
                    "ci_low": None if ci["ci_low"] is None else ci["ci_low"] * 100,
                    "ci_high": None if ci["ci_high"] is None else ci["ci_high"] * 100,
                })
    return rows


def _type_of(rec: dict) -> str:
    types = set(rec["hallucination_types"] or [])
    if types == {"vision_hallucination"}:
        return "vision_only"
    if types == {"reasoning_hallucination"}:
        return "reasoning_only"
    if types >= {"vision_hallucination", "reasoning_hallucination"}:
        return "both"
    return "unlabeled"


def module_c13(pool: dict[str, list[dict]]) -> None:
    header = ["model", "group", "value", "n", "pct", "ci_low", "ci_high"]
    write_csv(OUT / "c1_profile_dataset.csv", header, _profile_rows(pool, lambda r: r["dataset"], DATASETS))
    hb_cats = sorted({(r["metadata"].get("category") or "other") for m in MODELS for r in pool[m]
                      if r["dataset"] == "hallusionbench"})
    write_csv(OUT / "c1_profile_hb_category.csv", header,
              _profile_rows(pool, lambda r: (r["metadata"].get("category") or "other")
                            if r["dataset"] == "hallusionbench" else None, hb_cats))
    caps = sorted({(r["metadata"].get("capability") or "other") for m in MODELS for r in pool[m]
                   if r["dataset"] == "mmvet"})
    write_csv(OUT / "c1_profile_mmvet_cap.csv", header,
              _profile_rows(pool, lambda r: (r["metadata"].get("capability") or "other")
                            if r["dataset"] == "mmvet" else None, caps))
    write_csv(OUT / "c1_profile_types.csv", header,
              _profile_rows(pool, _type_of, ["vision_only", "reasoning_only", "both", "unlabeled"], h1_only=True))
    write_csv(OUT / "c1_profile_rating.csv", header,
              _profile_rows(pool, lambda r: r["rating"], list(range(7))))
    write_csv(OUT / "c1_profile_correct.csv", header,
              _profile_rows(pool, lambda r: "correct" if r["correct"] else "wrong", ["correct", "wrong"]))
    write_csv(OUT / "c1_profile_vilp_case.csv", header,
              _profile_rows(pool, lambda r: f"case{r['metadata'].get('case')}"
                            if r["dataset"] == "vilp" else None, ["case1", "case2", "case3"]))


def write_report(rate_rows: list[dict], pool: dict[str, list[dict]]) -> None:
    lines = [
        "# 模块 C1：LUH 画像与漏检归因报告",
        "",
        "## 分组规模",
    ]
    for model in MODELS:
        counts = {g: sum(1 for r in pool[model] if r["group"] == g)
                  for g in ("luh", "matched_neg", "detected_hallu")}
        lines.append(f"- {model}: {counts}")
    lines += [
        "",
        "## LUH 规模（每模型 × 每方法 × α，全体已评估样本合并，c1_luh_rate.csv）",
        "luh_share(α) = P(s_H1 ≤ Q_α(H0))，α ∈ {0.25, 0.50}；excess = share − α。",
    ]
    for row in rate_rows:
        if row["luh_share"] is None:
            lines.append(f"- {row['model']} / {row['method']} (α={row['alpha']:.2f}): unavailable (no valid samples)")
            continue
        lines.append(
            f"- {row['model']} / {row['method']} (α={row['alpha']:.2f}): "
            f"{row['luh_share']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]，"
            f"excess {row['excess']:+.3f}"
        )
    lines += [
        "",
        "## 关键画像数字",
        "- vision 幻觉且答案正确的 LUH 占比：见 c1_profile_types.csv 与 c1_profile_correct.csv 交叉项。",
        "- SE 簇收敛与 UMPIRE 分量对比：见 c1_se_clusters.csv / c1_umpire_components.csv。",
        "",
        "图表建议见 docs/实验一结果分析.md 各模块'可绘图'条目。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = load_pool()
    pooled_eval: dict[str, list[dict]] = {m: [] for m in MODELS}
    for model in MODELS:
        for dataset in DATASETS:
            pooled_eval[model].extend(evaluated(load_cell(model, dataset)))
    rate_rows = module_c11(pooled_eval)
    module_c12(pool)
    module_c13(pool)
    write_report(rate_rows, pool)
    print(f"module C1 done -> {OUT}")


if __name__ == "__main__":
    main()
