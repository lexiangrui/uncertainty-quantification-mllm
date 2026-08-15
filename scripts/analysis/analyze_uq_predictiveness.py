#!/usr/bin/env python3
"""Evaluate whether answer-only UQ scores predict correctness and hallucination.

The analysis joins the canonical split generation, UQ, and Judge outputs,
reports model x dataset x method metrics, and uses a group-id cluster bootstrap
for AUROC confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_jsonl_records


MODELS = ("llava", "qwen", "internvl")
DATASETS = ("vilp", "hallusionbench", "mmvet")
METHOD_COLUMNS = {
    "perplexity": "ppl",
    "semantic_entropy": "semantic_entropy",
    "umpire": "umpire",
}
TARGETS = ("error", "correct", "hallucination", "hallucination_given_error")


def jsonl_samples(path: Path) -> Iterable[dict]:
    for record in load_jsonl_records(path):
        if record.get("record_type") == "sample":
            yield record


def finite_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_rows(project_root: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    exclusions: list[dict] = []
    for model in MODELS:
        for dataset in DATASETS:
            base = project_root / "results"
            generation_path = base / "generation" / model / "greedy" / f"{dataset}.jsonl"
            uq_path = base / "uq" / model / f"{dataset}.jsonl"
            judge_path = base / "judging" / model / f"{dataset}.jsonl"

            generation_ids = {
                item["sample"]["sample_id"] for item in jsonl_samples(generation_path)
            }
            uq_by_id = {
                item["sample"]["sample_id"]: item for item in jsonl_samples(uq_path)
            }
            judge_by_id = {
                item["sample"]["sample_id"]: item for item in jsonl_samples(judge_path)
            }
            reasons: defaultdict[str, int] = defaultdict(int)
            included = 0
            for sample_id in sorted(generation_ids):
                judge_item = judge_by_id.get(sample_id)
                uq_item = uq_by_id.get(sample_id)
                if judge_item is None:
                    reasons["missing_judge_record"] += 1
                    continue
                judge = judge_item.get("judge")
                if not isinstance(judge, dict) or judge.get("valid") is not True:
                    error = judge.get("error") if isinstance(judge, dict) else None
                    if isinstance(error, str) and error.startswith(
                        "greedy response cannot be separated into three parts"
                    ):
                        reasons["xml_format_failure"] += 1
                    else:
                        reasons["invalid_judge"] += 1
                    continue
                if uq_item is None:
                    reasons["missing_uq_record"] += 1
                    continue
                uq = uq_item.get("uq", {})
                scores: dict[str, float] = {}
                invalid_method = None
                for method, column in METHOD_COLUMNS.items():
                    entry = uq.get(method)
                    score = finite_score(entry.get("score")) if isinstance(entry, dict) and entry.get("valid") is True else None
                    if score is None:
                        invalid_method = method
                        break
                    scores[column] = score
                if invalid_method is not None:
                    reasons[f"invalid_uq_{invalid_method}"] += 1
                    continue
                sample = judge_item["sample"]
                rows.append(
                    {
                        "sample_id": sample_id,
                        "group_id": sample.get("group_id", sample_id),
                        "model": model,
                        "dataset": dataset,
                        "correct": int(bool(judge["correct"])),
                        "error": int(not bool(judge["correct"])),
                        "hallucination": int(bool(judge["hallucination"])),
                        "rating": judge.get("rating"),
                        **scores,
                    }
                )
                included += 1
            exclusions.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "generated": len(generation_ids),
                    "included": included,
                    "excluded": len(generation_ids) - included,
                    "reasons": dict(sorted(reasons.items())),
                }
            )
    return rows, exclusions


def auroc(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        midrank = ((index + 1) + end) / 2.0
        rank_sum += midrank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    true_positives = 0
    previous_recall = 0.0
    result = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        true_positives += sum(label for _, label in ordered[index:end])
        recall = true_positives / positives
        precision = true_positives / end
        result += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return result


def rejection_area(scores: list[float], labels: list[int]) -> float:
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    total_positives = float(sum(labels))
    slot_mass: list[float] = []
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group_rate = sum(label for _, label in ordered[index:end]) / (end - index)
        slot_mass.extend([group_rate] * (end - index))
        index = end
    rejected_positives = 0.0
    values = []
    count = len(labels)
    for rejected in range(count):
        retained = count - rejected
        values.append(1.0 - (total_positives - rejected_positives) / retained)
        rejected_positives += slot_mass[rejected]
    return statistics.fmean(values)


def prr(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    if positives == 0 or positives == len(labels):
        return None
    area = rejection_area(scores, labels)
    random_area = 1.0 - positives / len(labels)
    oracle_area = rejection_area([float(label) for label in labels], labels)
    return (area - random_area) / (oracle_area - random_area)


def quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_ci(
    rows: list[dict],
    score_key: str,
    label_key: str,
    metric: Callable[[list[float], list[int]], float | None],
    samples: int,
    seed: int,
    score_sign: float = 1.0,
) -> tuple[float | None, float | None, int]:
    groups: defaultdict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    members = list(groups.values())
    generator = random.Random(seed)
    estimates: list[float] = []
    undefined = 0
    for _ in range(samples):
        replicate = [
            row
            for _index in range(len(members))
            for row in members[generator.randrange(len(members))]
        ]
        value = metric(
            [score_sign * row[score_key] for row in replicate],
            [row[label_key] for row in replicate],
        )
        if value is None:
            undefined += 1
        else:
            estimates.append(value)
    if not estimates:
        return None, None, undefined
    return quantile(estimates, 0.025), quantile(estimates, 0.975), undefined


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def tail_rates(scores: list[float], labels: list[int], fraction: float = 0.2) -> dict:
    count = max(1, math.ceil(len(scores) * fraction))
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    bottom = [labels[i] for i in order[:count]]
    top = [labels[i] for i in order[-count:]]
    prevalence = sum(labels) / len(labels)
    bottom_rate = sum(bottom) / len(bottom)
    top_rate = sum(top) / len(top)
    return {
        "tail_n": count,
        "bottom20_rate": bottom_rate,
        "top20_rate": top_rate,
        "top20_lift_vs_prevalence": top_rate / prevalence if prevalence else None,
        "top_vs_bottom_risk_ratio": top_rate / bottom_rate if bottom_rate else None,
        "top_minus_bottom": top_rate - bottom_rate,
    }


def evaluate(rows: list[dict], bootstrap_samples: int) -> list[dict]:
    results: list[dict] = []
    for model in MODELS:
        for dataset in DATASETS:
            cell = [row for row in rows if row["model"] == model and row["dataset"] == dataset]
            for method, score_key in METHOD_COLUMNS.items():
                for target in TARGETS:
                    subset = cell
                    label_key = target
                    score_sign = 1.0
                    if target == "correct":
                        score_sign = -1.0  # confidence: lower UQ means more likely correct
                    elif target == "hallucination_given_error":
                        subset = [row for row in cell if row["error"] == 1]
                        label_key = "hallucination"
                    scores = [score_sign * row[score_key] for row in subset]
                    labels = [row[label_key] for row in subset]
                    point_auroc = auroc(scores, labels) if subset else None
                    ci_low, ci_high, undefined = (None, None, 0)
                    if point_auroc is not None:
                        ci_low, ci_high, undefined = bootstrap_ci(
                            subset,
                            score_key,
                            label_key,
                            auroc,
                            bootstrap_samples,
                            stable_seed(model, dataset, method, target),
                            score_sign,
                        )
                    tails = tail_rates(scores, labels) if subset and 0 < sum(labels) < len(labels) else {}
                    results.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "method": method,
                            "target": target,
                            "score_direction": "confidence=-UQ" if target == "correct" else "risk=UQ",
                            "n": len(subset),
                            "positive_n": sum(labels),
                            "positive_rate": sum(labels) / len(labels) if labels else None,
                            "auroc": point_auroc,
                            "auroc_ci_low": ci_low,
                            "auroc_ci_high": ci_high,
                            "bootstrap_undefined": undefined,
                            "auprc": average_precision(scores, labels) if subset else None,
                            "auprc_random_baseline": sum(labels) / len(labels) if labels else None,
                            "prr": prr(scores, labels) if subset else None,
                            "rank_biserial": None if point_auroc is None else 2.0 * point_auroc - 1.0,
                            **tails,
                        }
                    )
    return results


def label_relationship(rows: list[dict]) -> list[dict]:
    output = []
    for model in MODELS:
        for dataset in DATASETS:
            cell = [row for row in rows if row["model"] == model and row["dataset"] == dataset]
            counts = {(correct, hall): 0 for correct in (0, 1) for hall in (0, 1)}
            for row in cell:
                counts[(row["correct"], row["hallucination"])] += 1
            a = counts[(0, 1)]
            b = counts[(0, 0)]
            c = counts[(1, 1)]
            d = counts[(1, 0)]
            denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
            corrected = [value + 0.5 for value in (a, b, c, d)]
            output.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "n": len(cell),
                    "accuracy": (c + d) / len(cell),
                    "hallucination_rate": (a + c) / len(cell),
                    "wrong_hallucination_n": a,
                    "wrong_clean_n": b,
                    "correct_hallucination_n": c,
                    "correct_clean_n": d,
                    "hallucination_rate_given_wrong": a / (a + b) if a + b else None,
                    "hallucination_rate_given_correct": c / (c + d) if c + d else None,
                    "phi_error_hallucination": (a * d - b * c) / denominator if denominator else None,
                    "odds_ratio_error_hallucination": corrected[0] * corrected[3] / (corrected[1] * corrected[2]),
                }
            )
    return output


def percentile_ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[index]]:
            end += 1
        percentile = ((index + end - 1) / 2.0 + 0.5) / len(values)
        for position in ordered[index:end]:
            ranks[position] = percentile
        index = end
    return ranks


def risk_deciles(rows: list[dict]) -> list[dict]:
    buckets: defaultdict[tuple[str, str, int], list[int]] = defaultdict(list)
    for model in MODELS:
        for dataset in DATASETS:
            cell = [row for row in rows if row["model"] == model and row["dataset"] == dataset]
            for method, score_key in METHOD_COLUMNS.items():
                ranks = percentile_ranks([row[score_key] for row in cell])
                for row, rank in zip(cell, ranks):
                    decile = min(10, int(rank * 10) + 1)
                    for target in ("error", "hallucination"):
                        buckets[(method, target, decile)].append(row[target])
    output = []
    for method in METHOD_COLUMNS:
        for target in ("error", "hallucination"):
            for decile in range(1, 11):
                labels = buckets[(method, target, decile)]
                output.append(
                    {
                        "method": method,
                        "target": target,
                        "uq_decile": decile,
                        "n": len(labels),
                        "positive_rate": sum(labels) / len(labels) if labels else None,
                    }
                )
    return output


def fmt(value: object, digits: int = 3) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def macro_summary(metrics: list[dict]) -> list[dict]:
    output = []
    for target in TARGETS:
        for method in METHOD_COLUMNS:
            cells = [row for row in metrics if row["target"] == target and row["method"] == method and row["auroc"] is not None]
            values = [row["auroc"] for row in cells]
            output.append(
                {
                    "target": target,
                    "method": method,
                    "cells": len(cells),
                    "macro_auroc_mean": statistics.fmean(values) if values else None,
                    "macro_auroc_median": statistics.median(values) if values else None,
                    "min_auroc": min(values) if values else None,
                    "max_auroc": max(values) if values else None,
                    "cells_auroc_above_0_5": sum(value > 0.5 for value in values),
                    "cells_ci_above_0_5": sum(row["auroc_ci_low"] is not None and row["auroc_ci_low"] > 0.5 for row in cells),
                    "mean_auprc_lift": statistics.fmean(
                        row["auprc"] / row["auprc_random_baseline"]
                        for row in cells
                        if row["auprc_random_baseline"] not in (None, 0)
                    ) if cells else None,
                    "mean_prr": statistics.fmean(row["prr"] for row in cells if row["prr"] is not None) if cells else None,
                    "mean_top_minus_bottom": statistics.fmean(row["top_minus_bottom"] for row in cells if row.get("top_minus_bottom") is not None) if cells else None,
                }
            )
    return output


def write_svg_heatmap(path: Path, metrics: list[dict]) -> None:
    cells = [(model, dataset) for model in MODELS for dataset in DATASETS]
    methods = list(METHOD_COLUMNS)
    targets = ("error", "hallucination")
    lookup = {(r["model"], r["dataset"], r["method"], r["target"]): r for r in metrics}
    width, height = 1180, 440
    left, top, cell_w, cell_h = 155, 80, 105, 45
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.small{font-size:11px}.label{font-size:13px}.value{font-size:13px;font-weight:600}</style>',
        '<text x="20" y="28" font-size="18" font-weight="600">UQ discrimination AUROC (higher is better; chance = 0.50)</text>',
    ]
    for column, (model, dataset) in enumerate(cells):
        x = left + column * cell_w + cell_w / 2
        pieces.append(f'<text class="small" text-anchor="middle" x="{x}" y="50">{model}</text>')
        pieces.append(f'<text class="small" text-anchor="middle" x="{x}" y="66">{dataset.replace("hallusionbench", "hallusion")}</text>')
    row_index = 0
    for target in targets:
        pieces.append(f'<text class="label" x="15" y="{top + row_index * cell_h + 29}">{"Error detection" if target == "error" else "Hallucination detection"}</text>')
        row_index += 1
        for method in methods:
            y = top + row_index * cell_h
            pieces.append(f'<text class="small" text-anchor="end" x="145" y="{y + 28}">{method}</text>')
            for column, (model, dataset) in enumerate(cells):
                value = lookup[(model, dataset, method, target)]["auroc"]
                distance = max(-0.2, min(0.2, value - 0.5)) / 0.2
                if distance >= 0:
                    red = int(241 - 100 * distance); green = int(248 - 50 * distance); blue = int(233 - 120 * distance)
                else:
                    red = int(241 + 10 * distance); green = int(248 + 70 * distance); blue = int(233 + 20 * distance)
                x = left + column * cell_w
                pieces.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cell_w - 4}" height="{cell_h - 4}" rx="3" fill="rgb({red},{green},{blue})"/>')
                pieces.append(f'<text class="value" text-anchor="middle" x="{x + cell_w / 2}" y="{y + 29}">{value:.3f}</text>')
            row_index += 1
        row_index += 0.35
    pieces.append('</svg>')
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def write_svg_deciles(path: Path, deciles: list[dict]) -> None:
    lookup = {(r["method"], r["target"], r["uq_decile"]): r["positive_rate"] for r in deciles}
    colors = {"perplexity": "#4c78a8", "semantic_entropy": "#e45756", "umpire": "#54a24b"}
    width, height = 1080, 420
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202124}.small{font-size:12px}.title{font-size:16px;font-weight:600}</style>',
        '<text x="20" y="26" font-size="18" font-weight="600">Observed risk by within-cell UQ decile</text>',
    ]
    for panel, target in enumerate(("error", "hallucination")):
        x0, y0, plot_w, plot_h = 70 + panel * 525, 65, 440, 285
        rates = [lookup[(m, target, d)] for m in METHOD_COLUMNS for d in range(1, 11)]
        ymax = max(rates) * 1.08
        pieces.append(f'<text class="title" x="{x0}" y="48">{"Error rate" if target == "error" else "Hallucination rate"}</text>')
        for tick in range(5):
            value = ymax * tick / 4
            y = y0 + plot_h - plot_h * tick / 4
            pieces.append(f'<line x1="{x0}" y1="{y}" x2="{x0 + plot_w}" y2="{y}" stroke="#dddddd"/>')
            pieces.append(f'<text class="small" text-anchor="end" x="{x0 - 8}" y="{y + 4}">{value:.0%}</text>')
        for method in METHOD_COLUMNS:
            points = []
            for decile in range(1, 11):
                x = x0 + (decile - 1) * plot_w / 9
                y = y0 + plot_h - lookup[(method, target, decile)] / ymax * plot_h
                points.append(f"{x:.1f},{y:.1f}")
            pieces.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[method]}" stroke-width="2.5"/>')
            for point in points:
                x, y = point.split(',')
                pieces.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{colors[method]}"/>')
        for decile in range(1, 11):
            x = x0 + (decile - 1) * plot_w / 9
            pieces.append(f'<text class="small" text-anchor="middle" x="{x}" y="{y0 + plot_h + 22}">{decile}</text>')
        pieces.append(f'<text class="small" text-anchor="middle" x="{x0 + plot_w / 2}" y="{y0 + plot_h + 43}">UQ decile (low → high uncertainty)</text>')
    for index, method in enumerate(METHOD_COLUMNS):
        x = 345 + index * 170
        pieces.append(f'<line x1="{x}" y1="397" x2="{x + 24}" y2="397" stroke="{colors[method]}" stroke-width="3"/>')
        pieces.append(f'<text class="small" x="{x + 30}" y="401">{method}</text>')
    pieces.append('</svg>')
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def write_report(
    output: Path,
    rows: list[dict],
    exclusions: list[dict],
    metrics: list[dict],
    macros: list[dict],
    relationships: list[dict],
    bootstrap_samples: int,
) -> None:
    macro = {(r["target"], r["method"]): r for r in macros}
    lines = [
        "# Can uncertainty predict correctness and hallucination?",
        "",
        "## Analysis contract",
        "",
        f"- Valid joined main responses: {len(rows)}; excluded: {sum(r['excluded'] for r in exclusions)} of {sum(r['generated'] for r in exclusions)} generated responses.",
        "- Correctness: high UQ predicts error; equivalently, confidence = -UQ predicts correct with the same AUROC.",
        "- Hallucination: high UQ predicts the judge hallucination label (rating < 3).",
        "- Conditional specificity check: hallucination prediction is repeated only among incorrect answers.",
        f"- AUROC 95% CIs use {bootstrap_samples} cluster-bootstrap replicates at group_id level within each model x dataset cell.",
        "- Primary aggregation is the unweighted macro mean over the nine model x dataset cells; raw scores are never pooled across models.",
        "",
        "## Macro result over 9 model x dataset cells",
        "",
        "| Target | Method | Mean AUROC | Median | Range | >0.5 cells | CI >0.5 | Mean PRR | Top-bottom risk gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in ("error", "correct", "hallucination", "hallucination_given_error"):
        for method in METHOD_COLUMNS:
            value = macro[(target, method)]
            lines.append(
                f"| {target} | {method} | {fmt(value['macro_auroc_mean'])} | {fmt(value['macro_auroc_median'])} | "
                f"{fmt(value['min_auroc'])}–{fmt(value['max_auroc'])} | {value['cells_auroc_above_0_5']}/{value['cells']} | "
                f"{value['cells_ci_above_0_5']}/{value['cells']} | {fmt(value['mean_prr'])} | {fmt(value['mean_top_minus_bottom'])} |"
            )
    lines += [
        "",
        "## Base label rates",
        "",
        "| Model | Dataset | N | Accuracy | Hallucination | H given wrong | H given correct | phi(error,H) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in relationships:
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['n']} | {row['accuracy']:.1%} | {row['hallucination_rate']:.1%} | "
            f"{row['hallucination_rate_given_wrong']:.1%} | {row['hallucination_rate_given_correct']:.1%} | {fmt(row['phi_error_hallucination'])} |"
        )
    lines += [
        "",
        "## Main findings",
        "",
        "- The table above is generated from the current canonical K=10 results; no fixed historical values are embedded in this report.",
        "- Compare error, hallucination, and hallucination-given-error separately. A method that detects wrong answers does not necessarily detect hallucination within wrong answers.",
        "- Use the cell-level CSV and confidence intervals before interpreting a macro average as consistent across models and datasets.",
        "",
        "## Interpretation guardrails",
        "",
        "- Perplexity uses only the greedy final answer; semantic entropy and UMPIRE use ten sampled answers. All three therefore measure answer uncertainty, while hallucination is judged from vision/reasoning content. Hallucination AUROC is proxy validity, not a direct content-grounding score.",
        "- AUPRC must be compared with its cell-specific positive-rate baseline. The CSV keeps both values.",
        "- Top/bottom 20% rates are descriptive operating points selected without labels. They are not tuned deployment thresholds.",
        "- Judge labels are model-generated rather than human ground truth; the planned human audit remains necessary before a strong claim.",
        "",
        "## Artifacts",
        "",
        "- metrics_by_cell.csv: all target x model x dataset x method metrics and AUROC CIs",
        "- macro_summary.csv: nine-cell macro summaries",
        "- label_relationship.csv: correctness-hallucination relationship",
        "- risk_by_decile.csv: empirical error/hallucination rate by within-cell UQ decile",
        "- exclusions.json: exact inclusion accounting",
        "- auroc_heatmap.svg and risk_by_decile.svg: main figures",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, exclusions = load_rows(args.project_root)
    metrics = evaluate(rows, args.bootstrap_samples)
    macros = macro_summary(metrics)
    relationships = label_relationship(rows)
    deciles = risk_deciles(rows)

    write_csv(args.output_dir / "metrics_by_cell.csv", metrics)
    write_csv(args.output_dir / "macro_summary.csv", macros)
    write_csv(args.output_dir / "label_relationship.csv", relationships)
    write_csv(args.output_dir / "risk_by_decile.csv", deciles)
    (args.output_dir / "exclusions.json").write_text(
        json.dumps(exclusions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_svg_heatmap(args.output_dir / "auroc_heatmap.svg", metrics)
    write_svg_deciles(args.output_dir / "risk_by_decile.svg", deciles)
    write_report(args.output_dir, rows, exclusions, metrics, macros, relationships, args.bootstrap_samples)

    print(f"included={len(rows)} excluded={sum(row['excluded'] for row in exclusions)}")
    for target in ("error", "hallucination", "hallucination_given_error"):
        print(f"[{target}]")
        for method in METHOD_COLUMNS:
            row = next(item for item in macros if item["target"] == target and item["method"] == method)
            print(
                f"  {method:18} macro_AUROC={fmt(row['macro_auroc_mean'])} "
                f"range={fmt(row['min_auroc'])}-{fmt(row['max_auroc'])} "
                f"CI>0.5={row['cells_ci_above_0_5']}/{row['cells']}"
            )
    print(f"report={args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
