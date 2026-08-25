#!/usr/bin/env python3
"""Section 3.3: the low-uncertainty hallucination (LUH) blind zone.

The primary LUH scale follows the historical blind-zone script exactly.  For
each model and UQ method, pool the three datasets and define

    LUH_share(alpha) = P(s_{H=1} <= Q_alpha(s_{H=0})),

for alpha in {0.25, 0.50}.  Ties at the non-hallucination threshold are
included.  The script reports both the selected hallucination count and its
share among all hallucinations, with group-bootstrap confidence intervals.

The cell-wise bottom-20% calculations are retained only as attribution
diagnostics; they are not used as the LUH scale definition.

Attribution tables explain why each method misses: PPL vs answer length and
format, SE cluster collapse, UMPIRE component decomposition.

Outputs CSV tables under results/analysis/exp1/luh/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.exp1_common import (  # noqa: E402
    DATASETS,
    METHODS,
    MODELS,
    RESULTS,
    all3_valid,
    answer_format,
    bootstrap_reps,
    cell_percentiles,
    evaluated,
    load_cell,
    metric_ci,
    write_csv,
)

OUT = RESULTS / "analysis" / "exp1" / "luh"
LUH_ALPHAS = (0.25, 0.50)
DIAGNOSTIC_REGION_ALPHA = 0.20


def load_cells_all3() -> dict[tuple[str, str], list[dict]]:
    return {(m, d): all3_valid(evaluated(load_cell(m, d))) for m in MODELS for d in DATASETS}


def module_z1(cells) -> list[dict]:
    """Bottom-20% blind-zone statistics per method per cell."""
    rows = []
    for (model, dataset), records in cells.items():
        pcts = cell_percentiles(records)
        low_mask = {m: pcts[m] <= DIAGNOSTIC_REGION_ALPHA for m in METHODS}
        for method in METHODS:
            low_idx = np.flatnonzero(low_mask[method])
            high_idx = np.flatnonzero(~low_mask[method])
            hal = np.array([1 if r["hallucination"] else 0 for r in records])
            err = np.array([0 if r["correct"] else 1 for r in records])
            n_hallu = int(hal.sum())
            severe_idx = np.flatnonzero(np.logical_and.reduce([low_mask[m] for m in METHODS]))
            sub_recs_low = [records[i] for i in low_idx]
            sub_recs_severe = [records[i] for i in severe_idx]
            hal_low_rate = metric_ci(
                lambda s, l: float(l.mean()) if len(l) else None,
                hal[low_idx].astype(float),
                hal[low_idx],
                bootstrap_reps(sub_recs_low),
            )
            severe_hal_rate = metric_ci(
                lambda s, l: float(l.mean()) if len(l) else None,
                hal[severe_idx].astype(float),
                hal[severe_idx],
                bootstrap_reps(sub_recs_severe),
            )
            rows.append({
                "model": model, "dataset": dataset, "method": method,
                "n": len(records), "n_low": len(low_idx),
                "low_hallu_rate": float(hal[low_idx].mean()),
                "low_hallu_rate_ci_low": hal_low_rate["ci_low"],
                "low_hallu_rate_ci_high": hal_low_rate["ci_high"],
                "high_hallu_rate": float(hal[high_idx].mean()),
                "hallu_in_low_share": float(hal[low_idx].sum() / n_hallu) if n_hallu else None,
                "low_error_rate": float(err[low_idx].mean()),
                "n_severe_all3_low": len(severe_idx),
                "severe_hallu_rate": severe_hal_rate["value"],
                "severe_hallu_rate_ci_low": severe_hal_rate["ci_low"],
                "severe_hallu_rate_ci_high": severe_hal_rate["ci_high"],
            })
    write_csv(OUT / "z1_blind_zone.csv", list(rows[0].keys()), rows)
    return rows


def module_z2(cells) -> list[dict]:
    """Consensus bottom-20% region of all three methods: size and label mix."""
    rows = []
    for (model, dataset), records in cells.items():
        pcts = cell_percentiles(records)
        severe = np.flatnonzero(
            np.logical_and.reduce([pcts[m] <= DIAGNOSTIC_REGION_ALPHA for m in METHODS])
        )
        sub = [records[i] for i in severe]
        n = len(sub)
        n_h = sum(1 for r in sub if r["hallucination"])
        n_e = sum(1 for r in sub if not r["correct"])
        # strict LUH: hallucinated AND all three low AND (for contrast) wrong
        rows.append({
            "model": model, "dataset": dataset,
            "n_consensus_low": n,
            "hallu_rate": n_h / n if n else None,
            "error_rate": n_e / n if n else None,
            "n_luh_hallu_consensus": n_h,
            "n_luh_correct_consensus": sum(1 for r in sub if r["correct"] and r["hallucination"]),
        })
    write_csv(OUT / "z2_consensus_low.csv", list(rows[0].keys()), rows)
    return rows


def module_z3(cells) -> list[dict]:
    """LUH_share(alpha): share of hallucinations at or below Q_alpha of the H=0 distribution."""
    rows = []
    for model in MODELS:
        pooled: list[dict] = []
        for dataset in DATASETS:
            pooled += cells[(model, dataset)]
        for method in METHODS:
            values = np.array([r["scores"][method] for r in pooled], dtype=float)
            hal = np.array([1 if r["hallucination"] else 0 for r in pooled])
            for alpha in LUH_ALPHAS:
                def share(v: np.ndarray, h: np.ndarray) -> float | None:
                    neg = v[h == 0]
                    pos = v[h == 1]
                    if len(neg) == 0 or len(pos) == 0:
                        return None
                    thr = np.quantile(neg, alpha)
                    return float((pos <= thr).mean())

                non_hallu = values[hal == 0]
                hallu = values[hal == 1]
                threshold = float(np.quantile(non_hallu, alpha))
                n_luh = int((hallu <= threshold).sum())
                ci = metric_ci(share, values, hal, bootstrap_reps(pooled))
                rows.append({
                    "model": model, "method": method, "alpha": alpha,
                    "n": len(pooled),
                    "n_non_hallu": len(non_hallu),
                    "n_hallu": len(hallu),
                    "threshold": threshold,
                    "n_luh": n_luh,
                    "luh_share": ci["value"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                    "null": alpha,
                    "excess": None if ci["value"] is None else ci["value"] - alpha,
                })
    write_csv(OUT / "z3_luh_share.csv", list(rows[0].keys()), rows)
    return rows


def _groups_for(records: list[dict], method: str) -> dict[str, list[dict]]:
    """Split into low-UQ hallucinations / other hallucinations / non-hallucinations."""
    pcts = cell_percentiles(records)
    out = {"low_hallu": [], "other_hallu": [], "non_hallu": []}
    for i, r in enumerate(records):
        if r["hallucination"]:
            out[
                "low_hallu"
                if pcts[method][i] <= DIAGNOSTIC_REGION_ALPHA
                else "other_hallu"
            ].append(r)
        else:
            out["non_hallu"].append(r)
    return out


def module_z4(cells) -> None:
    """Failure attribution per method: what makes low-UQ hallucinations invisible."""
    ppl_rows, se_rows, ump_rows = [], [], []
    for (model, dataset), records in cells.items():
        for method in METHODS:
            groups = _groups_for(records, method)
            for name, sub in groups.items():
                if not sub:
                    continue
                base = {"model": model, "dataset": dataset, "method": method, "group": name, "n": len(sub)}
                if method == "perplexity":
                    tc = np.array([r["token_count"] or 0 for r in sub], dtype=float)
                    base.update({
                        "token_count_mean": float(tc.mean()),
                        "token_count_median": float(np.median(tc)),
                        "ppl_mean": float(np.mean([r["scores"]["perplexity"] for r in sub])),
                    })
                    for fmt in ("yes/no", "numeric", "short(<=3w)", "long"):
                        cnt = sum(1 for r in sub if answer_format(r["answer"]) == fmt)
                        base[f"fmt_{fmt}_pct"] = cnt / len(sub)
                    ppl_rows.append(base)
                elif method == "semantic_entropy":
                    nc = np.array([r["se_n_clusters"] or 0 for r in sub], dtype=float)
                    dm = np.array([r["se_dominant_mass"] or 0.0 for r in sub], dtype=float)
                    base.update({
                        "n_clusters_mean": float(nc.mean()),
                        "single_cluster_pct": float((nc == 1).mean()),
                        "dominant_mass_mean": float(dm.mean()),
                        "se_score_mean": float(np.mean([r["scores"]["semantic_entropy"] for r in sub])),
                    })
                    se_rows.append(base)
                else:
                    sv = np.array([r["ump_semantic_volume"] or 0.0 for r in sub], dtype=float)
                    ic = np.array([r["ump_incoherence_mean"] or 0.0 for r in sub], dtype=float)
                    base.update({
                        "semantic_volume_mean": float(sv.mean()),
                        "incoherence_mean_mean": float(ic.mean()),
                        "umpire_score_mean": float(np.mean([r["scores"]["umpire"] for r in sub])),
                    })
                    ump_rows.append(base)
    write_csv(OUT / "z4_ppl_attribution.csv", list(ppl_rows[0].keys()), ppl_rows)
    write_csv(OUT / "z4_se_attribution.csv", list(se_rows[0].keys()), se_rows)
    write_csv(OUT / "z4_umpire_attribution.csv", list(ump_rows[0].keys()), ump_rows)

    # rank correlation between PPL and answer token count among hallucinations
    corr_rows = []
    for model in MODELS:
        pooled: list[dict] = []
        for dataset in DATASETS:
            pooled += cells[(model, dataset)]
        hal = [r for r in pooled if r["hallucination"]]
        tc = np.array([r["token_count"] or 0 for r in hal], dtype=float)
        ppl = np.array([r["scores"]["perplexity"] for r in hal], dtype=float)
        rho, p = sps.spearmanr(tc, ppl)
        corr_rows.append({
            "model": model, "n_hallu": len(hal),
            "spearman_rho": float(rho), "p_value": float(p),
        })
    write_csv(OUT / "z4_ppl_length_corr.csv", list(corr_rows[0].keys()), corr_rows)


def module_z5(cells) -> None:
    """Composition of consensus-LUH samples vs other hallucinations (profile)."""
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            records = cells[(model, dataset)]
            pcts = cell_percentiles(records)
            low_all = np.logical_and.reduce(
                [pcts[m] <= DIAGNOSTIC_REGION_ALPHA for m in METHODS]
            )
            for name, mask, need_hallu in (
                ("consensus_luh", low_all & np.array([r["hallucination"] for r in records]), True),
                ("other_hallu", ~low_all & np.array([r["hallucination"] for r in records]), True),
            ):
                sub = [r for i, r in enumerate(records) if mask[i]]
                if not sub:
                    continue
                by_type = {"vision_hallucination": 0, "reasoning_hallucination": 0, "both": 0}
                for r in sub:
                    t = set(r["hallucination_types"] or [])
                    if len(t) >= 2:
                        by_type["both"] += 1
                    elif t:
                        by_type[next(iter(t))] += 1
                rows.append({
                    "model": model, "dataset": dataset, "group": name, "n": len(sub),
                    "correct_share": sum(1 for r in sub if r["correct"]) / len(sub),
                    "rating_le2_share": sum(1 for r in sub if (r["rating"] or 0) <= 2) / len(sub),
                    "vision_type_share": by_type["vision_hallucination"] / len(sub),
                    "reasoning_type_share": by_type["reasoning_hallucination"] / len(sub),
                    "both_type_share": by_type["both"] / len(sub),
                })
    write_csv(OUT / "z5_luh_profile.csv", list(rows[0].keys()), rows)


def write_report(z3_rows) -> None:
    lines = [
        "# 3.3 LUH 盲区汇总",
        "",
        "LUH_share(α) = P(s_H1 ≤ Q_α(H0))；每个模型合并三个数据集，",
        "分别报告 α=0.25 和 α=0.50。阈值处并列样本全部纳入。",
        "",
        "| 模型 | 方法 | α | H=1 总数 | LUH 数量 | LUH_share | 95% CI | 零假设 | excess |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in z3_rows:
        lines.append(
            f"| {r['model']} | {r['method']} | {r['alpha']:.2f} | "
            f"{r['n_hallu']} | {r['n_luh']} | {r['luh_share']:.3f} | "
            f"({r['ci_low']:.3f}, {r['ci_high']:.3f}) | {r['null']} | {r['excess']:+.3f} |"
        )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Experiment 1 LUH blind-zone statistics.")
    parser.add_argument(
        "--scale-only",
        action="store_true",
        help="only recompute the alpha=0.25/0.50 LUH scale table and report",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cells = load_cells_all3()
    if args.scale_only:
        z3_rows = module_z3(cells)
        write_report(z3_rows)
        print(f"luh scale done -> {OUT}")
        return

    z1_rows = module_z1(cells)
    z2_rows = module_z2(cells)
    z3_rows = module_z3(cells)
    module_z4(cells)
    module_z5(cells)
    write_report(z3_rows)
    print(f"luh blind-zone done -> {OUT}")


if __name__ == "__main__":
    main()
