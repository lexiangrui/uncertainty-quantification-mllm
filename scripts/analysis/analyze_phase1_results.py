#!/usr/bin/env python3
"""First-stage analysis for correctness, hallucination, and low-UQ hallucination.

The script consumes the generation/UQ/Judge JSONL files under ``results/``.
It keeps the existing analysis implementation as the source of truth for
AUROC/AUPRC/PRR and adds the low-uncertainty blind-spot analysis needed by the
first-stage research questions.  All comparisons are within model x dataset
cells; raw UQ scores are never pooled across cells.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.analyze_uq_predictiveness import (  # noqa: E402
    DATASETS,
    METHOD_COLUMNS,
    MODELS,
    auroc,
    bootstrap_ci,
    evaluate,
    label_relationship,
    load_rows,
    macro_summary,
    risk_deciles,
    stable_seed,
    write_csv,
    write_report as write_base_report,
    write_svg_deciles,
    write_svg_heatmap,
)


PRIMARY_LOW_QUANTILE = 0.20
LOW_QUANTILES = (0.10, 0.20, 0.30)
METHOD_ORDER = tuple(METHOD_COLUMNS)


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def _rate(rows: list[dict], key: str) -> float | None:
    return _mean(float(row[key]) for row in rows) if rows else None


def _cell_rows(rows: list[dict], model: str, dataset: str, method: str) -> list[dict]:
    return [
        row
        for row in rows
        if row["model"] == model and row["dataset"] == dataset
    ]


def _score_key(method: str) -> str:
    return METHOD_COLUMNS[method]


def _ordered_low_high(rows: list[dict], method: str, fraction: float) -> tuple[list[dict], list[dict]]:
    count = max(1, math.ceil(len(rows) * fraction))
    score_key = _score_key(method)
    ordered = sorted(rows, key=lambda row: (row[score_key], row["sample_id"]))
    return ordered[:count], ordered[-count:]


def _quantile_value(rows: list[dict], method: str, fraction: float) -> float | None:
    if not rows:
        return None
    score_key = _score_key(method)
    ordered = sorted(row[score_key] for row in rows)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _quadrant(row: dict) -> str:
    if row["correct"] and not row["hallucination"]:
        return "correct_clean"
    if row["correct"] and row["hallucination"]:
        return "correct_hallucination"
    if row["error"] and not row["hallucination"]:
        return "wrong_clean"
    return "wrong_hallucination"


def low_uq_summary(rows: list[dict]) -> list[dict]:
    """Compute label-blind low-UQ rates and hallucination blind-spot shares."""
    output: list[dict] = []
    for model in MODELS:
        for dataset in DATASETS:
            for method in METHOD_ORDER:
                cell = _cell_rows(rows, model, dataset, method)
                total_hallucinations = sum(row["hallucination"] for row in cell)
                total_errors = sum(row["error"] for row in cell)
                for fraction in LOW_QUANTILES:
                    low, high = _ordered_low_high(cell, method, fraction)
                    low_h = sum(row["hallucination"] for row in low)
                    high_h = sum(row["hallucination"] for row in high)
                    low_e = sum(row["error"] for row in low)
                    high_e = sum(row["error"] for row in high)
                    low_severe = sum(
                        row["hallucination"] and row["error"] for row in low
                    )
                    output.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "method": method,
                            "low_fraction": fraction,
                            "cell_n": len(cell),
                            "low_n": len(low),
                            "high_n": len(high),
                            "low_score_cutoff": _quantile_value(cell, method, fraction),
                            "high_score_cutoff": _quantile_value(cell, method, 1.0 - fraction),
                            "low_error_rate": low_e / len(low) if low else None,
                            "high_error_rate": high_e / len(high) if high else None,
                            "low_hallucination_rate": low_h / len(low) if low else None,
                            "high_hallucination_rate": high_h / len(high) if high else None,
                            "low_severe_hallucination_rate": low_severe / len(low) if low else None,
                            "low_vs_high_hallucination_gap": (
                                low_h / len(low) - high_h / len(high)
                                if low and high
                                else None
                            ),
                            "low_hallucination_n": low_h,
                            "low_error_n": low_e,
                            "low_hallucination_share": (
                                low_h / total_hallucinations
                                if total_hallucinations
                                else None
                            ),
                            "high_hallucination_recall": (
                                high_h / total_hallucinations
                                if total_hallucinations
                                else None
                            ),
                            "cell_hallucination_n": total_hallucinations,
                            "cell_error_n": total_errors,
                            "low_correct_clean_n": sum(_quadrant(row) == "correct_clean" for row in low),
                            "low_correct_hallucination_n": sum(_quadrant(row) == "correct_hallucination" for row in low),
                            "low_wrong_clean_n": sum(_quadrant(row) == "wrong_clean" for row in low),
                            "low_wrong_hallucination_n": sum(_quadrant(row) == "wrong_hallucination" for row in low),
                        }
                    )
    return output


def low_uq_samples(rows: list[dict], fraction: float = PRIMARY_LOW_QUANTILE) -> list[dict]:
    output: list[dict] = []
    for model in MODELS:
        for dataset in DATASETS:
            for method in METHOD_ORDER:
                cell = _cell_rows(rows, model, dataset, method)
                low, _ = _ordered_low_high(cell, method, fraction)
                for row in low:
                    output.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "method": method,
                            "low_fraction": fraction,
                            "sample_id": row["sample_id"],
                            "group_id": row["group_id"],
                            "uq_score": row[_score_key(method)],
                            "correct": row["correct"],
                            "error": row["error"],
                            "hallucination": row["hallucination"],
                            "quadrant": _quadrant(row),
                            "rating": row.get("rating"),
                        }
                    )
    return output


def method_overlap(rows: list[dict], fraction: float = PRIMARY_LOW_QUANTILE) -> list[dict]:
    """Summarize intersections of low-UQ memberships across the three methods."""
    output: list[dict] = []
    for model in MODELS:
        for dataset in DATASETS:
            cell_by_method = {
                method: _cell_rows(rows, model, dataset, method)
                for method in METHOD_ORDER
            }
            low_ids: dict[str, set[str]] = {}
            for method, cell in cell_by_method.items():
                low, _ = _ordered_low_high(cell, method, fraction)
                low_ids[method] = {row["sample_id"] for row in low}
            sample_rows = {
                row["sample_id"]: row
                for method in METHOD_ORDER
                for row in cell_by_method[method]
            }
            patterns: defaultdict[str, list[dict]] = defaultdict(list)
            for sample_id, row in sample_rows.items():
                pattern = "|".join(
                    f"{method}={'low' if sample_id in low_ids[method] else 'not_low'}"
                    for method in METHOD_ORDER
                )
                patterns[pattern].append(row)
            for pattern, pattern_rows in sorted(patterns.items()):
                output.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "low_fraction": fraction,
                        "pattern": pattern,
                        "n": len(pattern_rows),
                        "error_n": sum(row["error"] for row in pattern_rows),
                        "hallucination_n": sum(row["hallucination"] for row in pattern_rows),
                        "error_rate": _rate(pattern_rows, "error"),
                        "hallucination_rate": _rate(pattern_rows, "hallucination"),
                        "severe_low_uq_n": sum(
                            row["error"] and row["hallucination"] for row in pattern_rows
                        ),
                    }
                )
    return output


def _macro_low(summary: list[dict], fraction: float) -> list[dict]:
    output = []
    for method in METHOD_ORDER:
        cells = [
            row
            for row in summary
            if row["method"] == method and row["low_fraction"] == fraction
        ]
        output.append(
            {
                "method": method,
                "low_fraction": fraction,
                "cells": len(cells),
                "macro_low_hallucination_rate": _mean(
                    row["low_hallucination_rate"] for row in cells if row["low_hallucination_rate"] is not None
                ),
                "macro_high_hallucination_rate": _mean(
                    row["high_hallucination_rate"] for row in cells if row["high_hallucination_rate"] is not None
                ),
                "macro_low_hallucination_share": _mean(
                    row["low_hallucination_share"] for row in cells if row["low_hallucination_share"] is not None
                ),
                "macro_high_hallucination_recall": _mean(
                    row["high_hallucination_recall"] for row in cells if row["high_hallucination_recall"] is not None
                ),
                "macro_low_severe_hallucination_rate": _mean(
                    row["low_severe_hallucination_rate"] for row in cells if row["low_severe_hallucination_rate"] is not None
                ),
            }
        )
    return output


def write_svg_low_uq(path: Path, summary: list[dict]) -> None:
    fraction = PRIMARY_LOW_QUANTILE
    cells = [(model, dataset) for model in MODELS for dataset in DATASETS]
    lookup = {
        (row["model"], row["dataset"], row["method"]): row["low_hallucination_rate"]
        for row in summary
        if row["low_fraction"] == fraction
    }
    colors = {"perplexity": "#4c78a8", "semantic_entropy": "#e45756", "umpire": "#54a24b"}
    width, height = 1180, 430
    left, top, cell_w, cell_h = 150, 75, 105, 38
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.small{font-size:11px}.value{font-size:12px;font-weight:600}</style>',
        '<text x="20" y="28" font-size="18" font-weight="600">Hallucination rate in the lowest 20% UQ region</text>',
    ]
    for index, (model, dataset) in enumerate(cells):
        x = left + index * cell_w + cell_w / 2
        pieces.append(f'<text class="small" text-anchor="middle" x="{x}" y="49">{model}</text>')
        pieces.append(f'<text class="small" text-anchor="middle" x="{x}" y="64">{dataset.replace("hallusionbench", "hallusion")}</text>')
    for row_index, method in enumerate(METHOD_ORDER):
        y = top + row_index * cell_h
        pieces.append(f'<text class="small" text-anchor="end" x="140" y="{y + 24}">{method}</text>')
        for index, cell in enumerate(cells):
            value = lookup.get((*cell, method))
            x = left + index * cell_w
            fill = colors[method]
            opacity = 0.12 + 0.78 * (float(value) if value is not None else 0.0)
            pieces.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cell_w - 4}" height="{cell_h - 4}" rx="3" fill="{fill}" fill-opacity="{opacity:.3f}"/>')
            pieces.append(f'<text class="value" text-anchor="middle" x="{x + cell_w / 2}" y="{y + 25}">{_fmt(value, 1) if value is not None else "N/A"}</text>')
    pieces.append('<text class="small" x="150" y="205">Values are descriptive operating-point rates; the 20% threshold is selected without labels.</text>')
    pieces.append('</svg>')
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def write_phase1_report(
    output: Path,
    rows: list[dict],
    metrics: list[dict],
    macros: list[dict],
    relationships: list[dict],
    low_summary: list[dict],
    low_macro: list[dict],
    bootstrap_samples: int,
) -> None:
    lookup = {(row["target"], row["method"]): row for row in macros}
    low20 = {(row["method"], row["low_fraction"]): row for row in low_macro}
    lines = [
        "# 第一阶段结果分析报告",
        "",
        "本报告仅分析第一阶段的三个数据集、三个模型和三种UQ方法，不包含第二阶段改进方法。",
        "",
        "## 分析口径",
        "",
        f"- 有效 joined 样本数：{len(rows)}。",
        "- 正确性与幻觉使用独立Judge字段，错误标签为 `error = 1 - correct`。",
        "- 主比较在每个模型×数据集单元格内部完成，宏平均对9个单元格等权。",
        f"- AUROC置信区间使用group_id级cluster bootstrap，共{bootstrap_samples}次重采样。",
        "- 低不确定性区域在每个模型×数据集×方法内部取最低10%、20%、30%，不使用标签选择阈值。",
        "",
        "## 1. 正确性与幻觉的标签关系",
        "",
        "| 模型 | 数据集 | N | Accuracy | Hallucination | H given wrong | H given correct | phi |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in relationships:
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['n']} | {row['accuracy']:.1%} | "
            f"{row['hallucination_rate']:.1%} | {row['hallucination_rate_given_wrong']:.1%} | "
            f"{row['hallucination_rate_given_correct']:.1%} | {_fmt(row['phi_error_hallucination'])} |"
        )
    lines += [
        "",
        "## 2. UQ预测目标拆分",
        "",
        "| 目标 | 方法 | 宏平均AUROC | 中位数 | 范围 | CI下界>0.5单元格 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for target in ("error", "hallucination", "hallucination_given_error"):
        for method in METHOD_ORDER:
            row = lookup[(target, method)]
            lines.append(
                f"| {target} | {method} | {_fmt(row['macro_auroc_mean'])} | {_fmt(row['macro_auroc_median'])} | "
                f"{_fmt(row['min_auroc'])}–{_fmt(row['max_auroc'])} | {row['cells_ci_above_0_5']}/{row['cells']} |"
            )
    error_mean = _mean(lookup[("error", method)]["macro_auroc_mean"] for method in METHOD_ORDER)
    hallucination_mean = _mean(lookup[("hallucination", method)]["macro_auroc_mean"] for method in METHOD_ORDER)
    conditional_mean = _mean(lookup[("hallucination_given_error", method)]["macro_auroc_mean"] for method in METHOD_ORDER)
    lines += [
        "",
        f"按三种方法宏平均，错误检测AUROC为{_fmt(error_mean)}，幻觉检测AUROC为{_fmt(hallucination_mean)}，错误样本内幻觉检测AUROC为{_fmt(conditional_mean)}。这三个数应结合逐单元格热力图和置信区间解释，不能将错误检测结果直接等同于幻觉检测结果。",
        "",
        "## 3. 低不确定性幻觉盲区",
        "",
        "| 方法 | 低UQ比例 | 低UQ幻觉率 | 高UQ幻觉率 | 幻觉落入低UQ比例 | 高UQ幻觉召回率 | 严重LUH率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fraction in LOW_QUANTILES:
        for method in METHOD_ORDER:
            row = low20[(method, fraction)]
            lines.append(
                f"| {method} | {fraction:.0%} | {_fmt(row['macro_low_hallucination_rate'], 1)} | "
                f"{_fmt(row['macro_high_hallucination_rate'], 1)} | {_fmt(row['macro_low_hallucination_share'], 1)} | "
                f"{_fmt(row['macro_high_hallucination_recall'], 1)} | {_fmt(row['macro_low_severe_hallucination_rate'], 1)} |"
            )
    lines += [
        "",
        "低UQ幻觉率不为零，或较高比例的幻觉样本落入低UQ区域，即构成baseline低不确定性幻觉盲区的证据。该指标是描述性漏检分析，不应被解释为经过标签调参后的分类阈值性能。",
        "",
        "## 4. 输出文件",
        "",
        "- `metrics_by_cell.csv`：模型×数据集×方法×目标的AUROC、AUPRC、PRR和尾部风险指标。",
        "- `macro_summary.csv`：9个单元格上的宏平均汇总。",
        "- `label_relationship.csv`：正确性与幻觉的四象限关系。",
        "- `risk_by_decile.csv`：UQ十分位数上的错误率、幻觉率和条件幻觉率。",
        "- `low_uq_summary.csv`：最低10%、20%、30%区域的低UQ盲区统计。",
        "- `low_uq_samples.csv`：最低20%区域的样本级审计清单。",
        "- `method_overlap.csv`：三种方法低UQ区域的交集模式。",
        "- `exclusions.json`：generation、Judge、UQ的纳入排除记录。",
        "- `auroc_heatmap.svg`、`risk_by_decile.svg`、`low_uq_hallucination.svg`：主要图形。",
        "",
        "## 5. 解释边界",
        "",
        "- UQ分数衡量模型输出的不确定性，不等于幻觉概率。",
        "- Judge标签属于自动标注，LUH样本应进一步人工抽样核验。",
        "- 低UQ阈值只按分数分位数确定，未使用幻觉标签调节。",
        "- 不同方法的原始分数不进行跨方法数值比较，只比较其在各自单元格中的排序能力。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, exclusions = load_rows(args.project_root)
    metrics = evaluate(rows, args.bootstrap_samples)
    macros = macro_summary(metrics)
    relationships = label_relationship(rows)
    deciles = risk_deciles(rows)
    low_summary = low_uq_summary(rows)
    low_macro = _macro_low(low_summary, PRIMARY_LOW_QUANTILE)
    low_macro_all = [*low_macro]
    for fraction in LOW_QUANTILES:
        if fraction != PRIMARY_LOW_QUANTILE:
            low_macro_all.extend(_macro_low(low_summary, fraction))
    low_samples = low_uq_samples(rows)
    overlap = method_overlap(rows)

    write_csv(args.output_dir / "metrics_by_cell.csv", metrics)
    write_csv(args.output_dir / "macro_summary.csv", macros)
    write_csv(args.output_dir / "label_relationship.csv", relationships)
    write_csv(args.output_dir / "risk_by_decile.csv", deciles)
    write_csv(args.output_dir / "low_uq_summary.csv", low_summary)
    write_csv(args.output_dir / "low_uq_macro_summary.csv", low_macro_all)
    write_csv(args.output_dir / "low_uq_samples.csv", low_samples)
    write_csv(args.output_dir / "method_overlap.csv", overlap)
    (args.output_dir / "exclusions.json").write_text(
        json.dumps(exclusions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "models": MODELS,
                "datasets": DATASETS,
                "methods": METHOD_ORDER,
                "targets": ("error", "correct", "hallucination", "hallucination_given_error"),
                "low_quantiles": LOW_QUANTILES,
                "primary_low_quantile": PRIMARY_LOW_QUANTILE,
                "bootstrap_samples": args.bootstrap_samples,
                "input_project_root": str(args.project_root),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_svg_heatmap(args.output_dir / "auroc_heatmap.svg", metrics)
    write_svg_deciles(args.output_dir / "risk_by_decile.svg", deciles)
    write_svg_low_uq(args.output_dir / "low_uq_hallucination.svg", low_summary)
    write_phase1_report(
        args.output_dir,
        rows,
        metrics,
        macros,
        relationships,
        low_summary,
        low_macro_all,
        args.bootstrap_samples,
    )

    print(f"included={len(rows)} excluded={sum(row['excluded'] for row in exclusions)}")
    for target in ("error", "hallucination", "hallucination_given_error"):
        print(f"[{target}]")
        for method in METHOD_ORDER:
            row = next(item for item in macros if item["target"] == target and item["method"] == method)
            print(
                f"  {method:18} macro_AUROC={_fmt(row['macro_auroc_mean'])} "
                f"range={_fmt(row['min_auroc'])}-{_fmt(row['max_auroc'])} "
                f"CI>0.5={row['cells_ci_above_0_5']}/{row['cells']}"
            )
    print(f"report={args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
