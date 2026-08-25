#!/usr/bin/env python3
"""Section 3.1: descriptive statistics over the formal aligned labels.

Outputs CSV tables under results/analysis/exp1/descriptive/ plus report.md.
Point estimates are crosschecked against results/metrics (tolerance 1e-9).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.exp1_common import (  # noqa: E402
    DATASETS,
    MODELS,
    RESULTS,
    answer_format,
    bootstrap_reps,
    evaluated,
    fmt_ci,
    iter_jsonl,
    load_cell,
    metric_ci,
    percentile_ci,
    write_csv,
)

OUT = RESULTS / "analysis" / "exp1" / "descriptive"


def module_d1() -> list[dict]:
    """Sample-validity flow per cell: greedy -> sections_valid -> aligned judge -> UQ."""
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            n_greedy = n_valid = 0
            for row in iter_jsonl(RESULTS / "generation" / model / "greedy" / f"{dataset}.jsonl"):
                if row.get("record_type") != "sample":
                    continue
                n_greedy += 1
                n_valid += bool(row.get("greedy", {}).get("sections_valid"))
            n_judge = 0
            for row in iter_jsonl(RESULTS / "judging" / model / f"{dataset}.jsonl"):
                if row.get("record_type") == "sample":
                    n_judge += 1
            ev = evaluated(load_cell(model, dataset))
            row = {
                "model": model,
                "dataset": dataset,
                "n_generated": n_greedy,
                "n_sections_valid": n_valid,
                "n_aligned_judge": n_judge,
                "n_evaluated": len(ev),
            }
            rows.append(row)
    write_csv(OUT / "d1_validity_flow.csv", list(rows[0].keys()), rows)
    return rows


def _rate_ci(mask: np.ndarray, records: list[dict]) -> dict:
    reps = bootstrap_reps(records)
    return metric_ci(lambda s, l: float(l.mean()) if len(l) else None, mask.astype(float), mask, reps)


def module_d2() -> tuple[list[dict], list[dict]]:
    """Accuracy and hallucination rate with clustered bootstrap CIs, per cell and pooled."""
    stats_rows: list[dict] = []
    cross_rows: list[dict] = []

    def add_block(records: list[dict], scope: str, model: str = "all", dataset: str = "all") -> dict:
        correct = np.array([1 if r["correct"] else 0 for r in records])
        hallu = np.array([1 if r["hallucination"] else 0 for r in records])
        acc = _rate_ci(correct, records)
        hr = _rate_ci(hallu, records)
        row = {
            "scope": scope,
            "model": model,
            "dataset": dataset,
            "n": len(records),
            "accuracy": acc["value"],
            "acc_ci_low": acc["ci_low"],
            "acc_ci_high": acc["ci_high"],
            "hallu_rate": hr["value"],
            "hr_ci_low": hr["ci_low"],
            "hr_ci_high": hr["ci_high"],
        }
        stats_rows.append(row)
        return row

    for model in MODELS:
        for dataset in DATASETS:
            ev = evaluated(load_cell(model, dataset))
            add_block(ev, f"{model}/{dataset}", model, dataset)
            metrics = json.loads((RESULTS / "metrics" / model / f"{dataset}.json").read_text())
            for key, metric in (("accuracy", "accuracy"), ("hallu_rate", "hallucination_rate")):
                ref = metrics["labels"][metric]["value"]
                mine = next(r for r in stats_rows if r["scope"] == f"{model}/{dataset}")[key]
                cross_rows.append({
                    "scope": f"{model}/{dataset}", "metric": metric,
                    "value_analysis": mine, "value_metrics": ref, "abs_diff": abs(mine - ref),
                })
    for model in MODELS:
        pooled = []
        for dataset in DATASETS:
            pooled += evaluated(load_cell(model, dataset))
        add_block(pooled, f"{model}/all", model)
    grand = []
    for model in MODELS:
        for dataset in DATASETS:
            grand += evaluated(load_cell(model, dataset))
    add_block(grand, "all/all")

    write_csv(OUT / "d2_label_stats.csv", list(stats_rows[0].keys()), stats_rows)
    write_csv(
        OUT / "d2_crosscheck.csv",
        ["scope", "metric", "value_analysis", "value_metrics", "abs_diff"],
        cross_rows,
    )
    return stats_rows, cross_rows


JOINT = (
    ("correct_no_hallu", 1, 0),
    ("correct_with_hallu", 1, 1),
    ("wrong_no_hallu", 0, 0),
    ("wrong_with_hallu", 0, 1),
)


def module_d3() -> list[dict]:
    """C x H joint distribution per cell and pooled, plus conditional rates."""
    rows = []
    scopes = [(m, d) for m in MODELS for d in DATASETS]
    scopes += [(m, "all") for m in MODELS] + [("all", "all")]
    for model, dataset in scopes:
        records = []
        if dataset == "all" and model == "all":
            for m in MODELS:
                for d in DATASETS:
                    records += evaluated(load_cell(m, d))
        elif dataset == "all":
            for d in DATASETS:
                records += evaluated(load_cell(model, d))
        else:
            records = evaluated(load_cell(model, dataset))
        n = len(records)
        row = {"scope": f"{model}/{dataset}", "model": model, "dataset": dataset, "n": n}
        for name, c, h in JOINT:
            cnt = sum(
                1 for r in records
                if int(bool(r["correct"])) == c and int(bool(r["hallucination"])) == h
            )
            row[name] = cnt
            row[f"{name}_pct"] = cnt / n if n else None
        hallu = [r for r in records if r["hallucination"]]
        wrong = [r for r in records if not r["correct"]]
        row["p_h_given_correct"] = (
            sum(1 for r in records if r["correct"] and r["hallucination"]) /
            max(1, sum(1 for r in records if r["correct"]))
        )
        row["p_h_given_wrong"] = (
            sum(1 for r in wrong if r["hallucination"]) / max(1, len(wrong))
        )
        row["p_e_given_h"] = sum(1 for r in hallu if not r["correct"]) / max(1, len(hallu))
        row["hallu_in_correct_share"] = (
            sum(1 for r in records if r["correct"] and r["hallucination"]) / max(1, len(hallu))
        )
        rows.append(row)
    write_csv(OUT / "d3_c_h_joint.csv", list(rows[0].keys()), rows)
    # long format (one row per C x H combination) for plotting
    long_rows = []
    for r in rows:
        for name, c, h in JOINT:
            long_rows.append({
                "scope": r["scope"], "model": r["model"], "dataset": r["dataset"],
                "correct": c, "hallucination": h, "n": r[name],
            })
    write_csv(OUT / "d3_c_h_joint_long.csv", list(long_rows[0].keys()), long_rows)
    return rows


def module_d4() -> tuple[list[dict], list[dict]]:
    """Judge rating distribution and hallucination-type composition among H=1."""
    rating_rows, type_rows = [], []
    scopes = [(m, d) for m in MODELS for d in DATASETS] + [("all", "all")]
    for model, dataset in scopes:
        if dataset == "all":
            records = []
            for m in MODELS:
                for d in DATASETS:
                    records += evaluated(load_cell(m, d))
        else:
            records = evaluated(load_cell(model, dataset))
        n = len(records)
        rating_row = {"scope": f"{model}/{dataset}", "n": n}
        for rating in range(7):
            cnt = sum(1 for r in records if r["rating"] == rating)
            rating_row[f"rating_{rating}"] = cnt
            rating_row[f"rating_{rating}_pct"] = cnt / n if n else None
        rating_rows.append(rating_row)

        hallu = [r for r in records if r["hallucination"]]
        types = {"vision_hallucination": 0, "reasoning_hallucination": 0, "both": 0, "none_listed": 0}
        for r in hallu:
            t = set(r["hallucination_types"] or [])
            if len(t) == 0:
                types["none_listed"] += 1
            elif t == {"vision_hallucination"}:
                types["vision_hallucination"] += 1
            elif t == {"reasoning_hallucination"}:
                types["reasoning_hallucination"] += 1
            else:
                types["both"] += 1
        type_rows.append({
            "scope": f"{model}/{dataset}", "n_hallu": len(hallu), **types,
            "vision_any": types["vision_hallucination"] + types["both"],
            "reasoning_any": types["reasoning_hallucination"] + types["both"],
        })
    write_csv(OUT / "d4_rating_dist.csv", list(rating_rows[0].keys()), rating_rows)
    write_csv(OUT / "d4_hallu_types.csv", list(type_rows[0].keys()), type_rows)
    return rating_rows, type_rows


def module_d5() -> list[dict]:
    """Answer-format composition per cell (context for the PPL attribution)."""
    rows = []
    for model in MODELS:
        for dataset in DATASETS:
            ev = evaluated(load_cell(model, dataset))
            row = {"scope": f"{model}/{dataset}", "n": len(ev)}
            counts: dict[str, int] = {}
            for r in ev:
                counts[answer_format(r["answer"])] = counts.get(answer_format(r["answer"]), 0) + 1
            for key, cnt in sorted(counts.items()):
                row[key] = cnt
                row[f"{key}_pct"] = cnt / len(ev)
            rows.append(row)
    keys = sorted({k for r in rows for k in r if k not in {"scope", "n"} and not k.endswith("_pct")})
    header = ["scope", "n"] + keys + [f"{k}_pct" for k in keys]
    write_csv(OUT / "d5_answer_format.csv", header, rows)
    return rows


def write_report(stats_rows, joint_rows, type_rows, cross_rows) -> None:
    lines = ["# 3.1 描述性统计汇总", ""]
    lines.append("## 准确率与幻觉率（d2_label_stats.csv）")
    lines.append("| 单元格 | n | Accuracy | Hallu. Rate |")
    lines.append("|---|---:|---|---|")
    for r in stats_rows:
        lines.append(
            f"| {r['scope']} | {r['n']} | "
            f"{fmt_ci(r['accuracy'], r)} | {fmt_ci(r['hallu_rate'], r)} |"
        )
    lines += ["", "## C×H 联合分布（d3_c_h_joint.csv，计数）"]
    lines.append("| 单元格 | 正确无幻 | 正确含幻 | 错误无幻 | 错误含幻 | P(H|C=1) | P(H|C=0) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in joint_rows:
        lines.append(
            f"| {r['scope']} | {r['correct_no_hallu']} | {r['correct_with_hallu']} | "
            f"{r['wrong_no_hallu']} | {r['wrong_with_hallu']} | "
            f"{r['p_h_given_correct']:.3f} | {r['p_h_given_wrong']:.3f} |"
        )
    lines += ["", "## 幻觉类型构成（d4_hallu_types.csv）"]
    lines.append("| 单元格 | H=1 数 | vision | reasoning | both | 未列类型 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in type_rows:
        lines.append(
            f"| {r['scope']} | {r['n_hallu']} | {r['vision_hallucination']} | "
            f"{r['reasoning_hallucination']} | {r['both']} | {r['none_listed']} |"
        )
    diffs = [c["abs_diff"] for c in cross_rows]
    lines += [
        "",
        f"## 与 results/metrics 交叉核对：{len(diffs)} 项，最大绝对差 {max(diffs):.2e}。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    module_d1()
    stats_rows, cross_rows = module_d2()
    joint_rows = module_d3()
    _, type_rows = module_d4()
    module_d5()
    write_report(stats_rows, joint_rows, type_rows, cross_rows)
    bad = [c for c in cross_rows if c["abs_diff"] > 1e-9]
    print(f"descriptive done -> {OUT}")
    print(f"crosscheck vs results/metrics: {len(cross_rows) - len(bad)}/{len(cross_rows)} consistent")
    for c in bad:
        print("MISMATCH:", c)
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
