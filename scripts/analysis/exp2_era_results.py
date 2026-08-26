#!/usr/bin/env python3
"""Consolidated numerical analysis for Experiment 2 (ERA on LUH subsets).

This script intentionally produces no figures.  It joins the fixed per-model
LUH subsets with formal judge labels, the three baseline UQ scores, and ERA
attention components, then writes machine-readable tables plus a Chinese
results-only Markdown document.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.exp1_common import (  # noqa: E402
    DATASETS,
    METHODS,
    METHOD_SHORT,
    MODELS,
    RESULTS,
    bootstrap_reps,
    iter_jsonl,
    load_cell,
    write_csv,
)
from src.evaluation.metrics import auprc, auroc, prr  # noqa: E402

ERA_NAME = "era"
ALL_METHODS = (*METHODS, ERA_NAME)
DISPLAY_NAME = {**METHOD_SHORT, ERA_NAME: "ERA"}
METRIC_FNS: dict[str, Callable] = {"auroc": auroc, "auprc": auprc, "prr": prr}
DEFAULT_LAYERS = (0, 1)


def percentile_interval(
    values: Iterable[float], confidence: float = 0.95
) -> tuple[float, float]:
    """Return a finite-value percentile interval."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - confidence) * 50.0
    low, high = np.percentile(finite, (tail, 100.0 - tail))
    return float(low), float(high)


def two_sided_bootstrap_p(values: Iterable[float]) -> float:
    """Two-sided sign probability for paired bootstrap differences."""
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return float("nan")
    lower = float(np.mean(finite <= 0.0))
    upper = float(np.mean(finite >= 0.0))
    return min(1.0, 2.0 * min(lower, upper))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm family-wise adjustment, preserving NaN positions."""
    out = [float("nan")] * len(p_values)
    valid = [(i, float(p)) for i, p in enumerate(p_values) if np.isfinite(p)]
    valid.sort(key=lambda item: item[1])
    running = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(valid):
        running = max(running, min(1.0, (count - rank) * value))
        out[index] = running
    return out


def _normalized_answer_masses(payload: dict, layer: int) -> tuple[float, ...]:
    masses = payload["layer_masses"][str(layer)]
    heads = int(payload["n_heads"])
    n_answer = int(payload["section_tokens"]["answer"])
    if heads <= 0 or n_answer <= 0 or len(masses) < 3 or len(masses[2]) != 5:
        raise ValueError("invalid ERA attention payload")
    scale = heads * n_answer
    return tuple(float(value) / scale for value in masses[2])


def attention_features(payload: dict, layers: Sequence[int]) -> dict[str, float]:
    """Aggregate answer-to-region attention and ERA ratios over selected layers."""
    available = {int(layer) for layer in payload.get("layer_masses", {})}
    selected = [int(layer) for layer in layers if int(layer) in available]
    if not selected:
        raise ValueError(
            f"selected layers are absent; requested={list(layers)}, available={sorted(available)}"
        )
    layer_values = [_normalized_answer_masses(payload, layer) for layer in selected]
    means = np.asarray(layer_values, dtype=float).mean(axis=0)
    ratios, ratios_with_answer = [], []
    for image, prompt, vision, reasoning, answer in layer_values:
        internal = vision + reasoning
        prior = image + prompt + internal
        ratios.append(internal / prior if prior > 0 else float("nan"))
        with_answer = prior + answer
        ratios_with_answer.append(
            internal / with_answer if with_answer > 0 else float("nan")
        )
    image, prompt, vision, reasoning, answer = means.tolist()
    return {
        "attn_image": image,
        "attn_prompt_text": prompt,
        "attn_vision": vision,
        "attn_reasoning": reasoning,
        "attn_answer": answer,
        "attn_external": image + prompt,
        "attn_internal": vision + reasoning,
        "U_ERA": float(np.nanmean(ratios)),
        "U_ERA_with_answer": float(np.nanmean(ratios_with_answer)),
        "n_visual_tokens": int(payload["n_visual_tokens"]),
        "n_answer_tokens": int(payload["section_tokens"]["answer"]),
        "n_rationale_tokens": int(
            payload["section_tokens"]["vision"]
            + payload["section_tokens"]["reasoning"]
        ),
    }


def component_variant_scores(payload: dict, layers: Sequence[int]) -> dict[str, float]:
    """Canonical and V/R component-ablation scores, averaged layer-wise."""
    available = {int(layer) for layer in payload.get("layer_masses", {})}
    selected = [int(layer) for layer in layers if int(layer) in available]
    if not selected:
        raise ValueError("selected layers are absent")
    values: dict[str, list[float]] = {
        "vision_reasoning": [],
        "vision_only": [],
        "reasoning_only": [],
    }
    for layer in selected:
        image, prompt, vision, reasoning, _ = _normalized_answer_masses(payload, layer)
        external = image + prompt
        values["vision_reasoning"].append(
            (vision + reasoning) / (external + vision + reasoning)
        )
        values["vision_only"].append(vision / (external + vision))
        values["reasoning_only"].append(reasoning / (external + reasoning))
    return {name: float(np.mean(scores)) for name, scores in values.items()}


def _average_rank_percentile(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="stable")
    sorted_values = array[order]
    ranks = np.empty(array.size, dtype=float)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    return ranks / array.size


def _load_components(components_dir: Path, model: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for dataset in DATASETS:
        path = components_dir / model / f"{dataset}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for row in iter_jsonl(path):
            if row.get("record_type") != "sample":
                continue
            sample_id = row.get("sample", {}).get("sample_id")
            payload = row.get("era")
            if not sample_id or not payload:
                continue
            if sample_id in out:
                raise ValueError(f"{model}: duplicate ERA sample_id {sample_id}")
            out[sample_id] = payload
    return out


def load_experiment_data(
    subset_path: Path, components_dir: Path, layers: Sequence[int]
) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    subsets = json.loads(subset_path.read_text(encoding="utf-8"))
    by_model: dict[str, list[dict]] = {}
    for model in MODELS:
        if model not in subsets:
            raise ValueError(f"subset has no model {model}")
        positive_ids = list(subsets[model]["positive_ids"])
        negative_ids = list(subsets[model]["negative_ids"])
        if len(positive_ids) != 200 or len(negative_ids) != 200:
            raise ValueError(
                f"{model}: expected 200 positive and 200 negative IDs, got "
                f"{len(positive_ids)} and {len(negative_ids)}"
            )
        labels = {sample_id: 1 for sample_id in positive_ids}
        labels.update({sample_id: 0 for sample_id in negative_ids})
        if len(labels) != 400:
            raise ValueError(f"{model}: subset IDs are not unique")

        joined: dict[str, dict] = {}
        for dataset in DATASETS:
            for record in load_cell(model, dataset):
                if record["sample_id"] in labels:
                    item = dict(record)
                    item["bootstrap_group"] = f"{dataset}:{record['group_id']}"
                    joined[record["sample_id"]] = item
        components = _load_components(components_dir, model)
        missing = sorted(set(labels) - joined.keys())
        missing_era = sorted(set(labels) - components.keys())
        if missing or missing_era:
            raise ValueError(
                f"{model}: incomplete inputs: joined_missing={len(missing)}, "
                f"era_missing={len(missing_era)}"
            )

        records = []
        for sample_id in positive_ids + negative_ids:
            record = joined[sample_id]
            label = labels[sample_id]
            if bool(record["hallucination"]) != bool(label):
                raise ValueError(f"{model}/{sample_id}: subset and judge labels disagree")
            if any(record["scores"].get(method) is None for method in METHODS):
                raise ValueError(f"{model}/{sample_id}: invalid baseline score")
            features = attention_features(components[sample_id], layers)
            record.update(features)
            record["label"] = label
            record["era_payload"] = components[sample_id]
            record["pair_index"] = (
                positive_ids.index(sample_id)
                if label
                else negative_ids.index(sample_id)
            )
            records.append(record)
        by_model[model] = records
    return by_model, subsets


def _metric_bundle(
    records: list[dict], scores: np.ndarray, n_bootstrap: int
) -> tuple[dict[str, float], dict[str, list[float]]]:
    labels = np.asarray([record["label"] for record in records], dtype=int)
    reps = bootstrap_reps(records, n_bootstrap=n_bootstrap)
    point: dict[str, float] = {}
    boot: dict[str, list[float]] = {metric: [] for metric in METRIC_FNS}
    for metric, fn in METRIC_FNS.items():
        value = fn(scores, labels)
        point[metric] = float(value) if value is not None else float("nan")
        for indices in reps:
            replicate = fn(scores[indices], labels[indices])
            boot[metric].append(
                float(replicate) if replicate is not None else float("nan")
            )
    return point, boot


def performance_tables(
    by_model: dict[str, list[dict]], n_bootstrap: int, out_dir: Path
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    cached: dict[tuple[str, str], tuple[dict, dict]] = {}
    for model, records in by_model.items():
        for method in ALL_METHODS:
            scores = np.asarray(
                [
                    record["U_ERA"]
                    if method == ERA_NAME
                    else record["scores"][method]
                    for record in records
                ],
                dtype=float,
            )
            point, boot = _metric_bundle(records, scores, n_bootstrap)
            cached[(model, method)] = point, boot
            row = {"model": model, "method": method, "n": len(records), "n_pos": 200}
            for metric in METRIC_FNS:
                low, high = percentile_interval(boot[metric])
                row[metric] = point[metric]
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            rows.append(row)
    write_csv(out_dir / "era_vs_baselines.csv", list(rows[0]), rows)

    improvement: list[dict] = []
    for model in MODELS:
        for metric in METRIC_FNS:
            baseline = max(METHODS, key=lambda name: cached[(model, name)][0][metric])
            era_point, era_boot = cached[(model, ERA_NAME)]
            base_point, base_boot = cached[(model, baseline)]
            deltas = np.asarray(era_boot[metric]) - np.asarray(base_boot[metric])
            low, high = percentile_interval(deltas)
            improvement.append(
                {
                    "model": model,
                    "metric": metric,
                    "best_baseline": baseline,
                    "baseline_value": base_point[metric],
                    "era_value": era_point[metric],
                    "absolute_improvement": era_point[metric] - base_point[metric],
                    "relative_improvement_pct": (
                        (era_point[metric] - base_point[metric]) / abs(base_point[metric]) * 100
                        if base_point[metric] != 0
                        else float("nan")
                    ),
                    "ci_low": low,
                    "ci_high": high,
                    "p_bootstrap": two_sided_bootstrap_p(deltas),
                }
            )
    adjusted = holm_adjust([row["p_bootstrap"] for row in improvement])
    for row, value in zip(improvement, adjusted, strict=True):
        row["p_holm"] = value
    write_csv(out_dir / "era_improvement.csv", list(improvement[0]), improvement)
    return rows, improvement


def distribution_tables(by_model: dict[str, list[dict]], out_dir: Path) -> tuple[list[dict], list[dict]]:
    sample_rows, summary = [], []
    for model, records in by_model.items():
        for record in records:
            sample_rows.append(
                {
                    "model": model,
                    "dataset": record["dataset"],
                    "sample_id": record["sample_id"],
                    "group": "LUH" if record["label"] else "non_hallucination",
                    "label": record["label"],
                    "correct": int(bool(record["correct"])),
                    "era": record["U_ERA"],
                    "ppl": record["scores"]["perplexity"],
                    "se": record["scores"]["semantic_entropy"],
                    "umpire": record["scores"]["umpire"],
                    "attn_image": record["attn_image"],
                    "attn_prompt_text": record["attn_prompt_text"],
                    "attn_vision": record["attn_vision"],
                    "attn_reasoning": record["attn_reasoning"],
                    "attn_external": record["attn_external"],
                    "attn_internal": record["attn_internal"],
                }
            )
        for label, group in ((1, "LUH"), (0, "non_hallucination")):
            values = np.asarray([record["U_ERA"] for record in records if record["label"] == label])
            summary.append(
                {
                    "model": model,
                    "group": group,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "median": float(np.median(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    write_csv(out_dir / "era_sample_scores.csv", list(sample_rows[0]), sample_rows)
    write_csv(out_dir / "era_distribution_summary.csv", list(summary[0]), summary)

    consistency = []
    for model, records in by_model.items():
        labels = np.asarray([record["label"] for record in records], dtype=int)
        scores = np.asarray([record["U_ERA"] for record in records])
        pos, neg = scores[labels == 1], scores[labels == 0]
        auc = auroc(scores, labels)
        consistency.append(
            {
                "model": model,
                "n": len(records),
                "era_auroc": auc,
                "cliffs_delta": 2 * auc - 1 if auc is not None else float("nan"),
                "luh_median": float(np.median(pos)),
                "non_hallucination_median": float(np.median(neg)),
                "median_gap": float(np.median(pos) - np.median(neg)),
                "direction_consistent": bool(np.median(pos) > np.median(neg)),
            }
        )
    write_csv(out_dir / "cross_model_consistency.csv", list(consistency[0]), consistency)
    return summary, consistency


def correctness_stratified_ablation(
    by_model: dict[str, list[dict]], n_bootstrap: int, out_dir: Path
) -> list[dict]:
    """Evaluate ERA within correct and incorrect final-answer strata."""
    rows: list[dict] = []
    for model, records in by_model.items():
        for correct, stratum in ((True, "correct"), (False, "incorrect")):
            selected = [record for record in records if bool(record["correct"]) == correct]
            labels = np.asarray([record["label"] for record in selected], dtype=int)
            n_hallucination = int(labels.sum())
            n_non_hallucination = int(labels.size - n_hallucination)
            if not selected or n_hallucination == 0 or n_non_hallucination == 0:
                raise ValueError(
                    f"{model}/{stratum}: ERA evaluation requires both hallucination classes; "
                    f"got n_hallucination={n_hallucination}, "
                    f"n_non_hallucination={n_non_hallucination}"
                )
            scores = np.asarray([record["U_ERA"] for record in selected], dtype=float)
            point, boot = _metric_bundle(selected, scores, n_bootstrap)
            row = {
                "model": model,
                "stratum": stratum,
                "correct": int(correct),
                "n": len(selected),
                "n_hallucination": n_hallucination,
                "n_non_hallucination": n_non_hallucination,
            }
            for metric in METRIC_FNS:
                low, high = percentile_interval(boot[metric])
                row[metric] = point[metric]
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
                row[f"{metric}_bootstrap_valid"] = int(
                    np.isfinite(np.asarray(boot[metric], dtype=float)).sum()
                )
            rows.append(row)
    write_csv(out_dir / "correctness_stratified_ablation.csv", list(rows[0]), rows)
    return rows


def _layer_windows(available: Sequence[int]) -> dict[str, tuple[int, ...]]:
    layers = tuple(sorted(available))
    candidates: list[tuple[str, tuple[int, ...]]] = []
    if 0 in layers:
        candidates.append(("Layer 0", (0,)))
    if 1 in layers:
        candidates.append(("Layer 1", (1,)))
    early2 = tuple(layer for layer in (0, 1) if layer in layers)
    if early2:
        candidates.append(("Layers 0-1", early2))
    early4 = tuple(layer for layer in range(4) if layer in layers)
    if len(early4) >= 2:
        candidates.append(("Layers 0-3", early4))
    if len(layers) >= 2:
        middle = len(layers) // 2
        candidates.append(("Middle 2", layers[max(0, middle - 1) : middle + 1]))
        candidates.append(("Final 2", layers[-2:]))
    candidates.append(("All layers", layers))
    out: dict[str, tuple[int, ...]] = {}
    seen: set[tuple[int, ...]] = set()
    for name, selection in candidates:
        if selection and selection not in seen:
            out[name] = selection
            seen.add(selection)
    return out


def ablation_tables(by_model: dict[str, list[dict]], n_bootstrap: int, out_dir: Path) -> tuple[list[dict], list[dict]]:
    window_rows, sweep_rows, component_rows = [], [], []
    for model, records in by_model.items():
        available = set.intersection(
            *[
                {int(layer) for layer in record["era_payload"]["layer_masses"]}
                for record in records
            ]
        )
        for name, layers in _layer_windows(sorted(available)).items():
            scores = np.asarray(
                [attention_features(record["era_payload"], layers)["U_ERA"] for record in records]
            )
            point, boot = _metric_bundle(records, scores, n_bootstrap)
            row = {"model": model, "layer_set": name, "layers": ",".join(map(str, layers))}
            for metric in METRIC_FNS:
                low, high = percentile_interval(boot[metric])
                row[metric] = point[metric]
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            window_rows.append(row)
        for layer in sorted(available):
            scores = np.asarray(
                [attention_features(record["era_payload"], [layer])["U_ERA"] for record in records]
            )
            point, _ = _metric_bundle(records, scores, n_bootstrap)
            sweep_rows.append({"model": model, "layer": layer, **point})

        variants = {
            record["sample_id"]: component_variant_scores(record["era_payload"], DEFAULT_LAYERS)
            for record in records
        }
        for variant in ("vision_reasoning", "vision_only", "reasoning_only"):
            scores = np.asarray([variants[record["sample_id"]][variant] for record in records])
            point, boot = _metric_bundle(records, scores, n_bootstrap)
            row = {"model": model, "variant": variant}
            for metric in METRIC_FNS:
                low, high = percentile_interval(boot[metric])
                row[metric] = point[metric]
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            component_rows.append(row)
    write_csv(out_dir / "decoder_layer_ablation.csv", list(window_rows[0]), window_rows)
    write_csv(out_dir / "decoder_layer_sweep.csv", list(sweep_rows[0]), sweep_rows)
    write_csv(out_dir / "component_ablation.csv", list(component_rows[0]), component_rows)
    return window_rows, component_rows


def select_cases(by_model: dict[str, list[dict]], out_dir: Path) -> list[dict]:
    candidates = []
    for model, records in by_model.items():
        by_label_index = {(record["label"], record["pair_index"]): record for record in records}
        percentiles = {}
        for name in (*METHODS, ERA_NAME):
            values = [
                record["U_ERA"] if name == ERA_NAME else record["scores"][name]
                for record in records
            ]
            ranked = _average_rank_percentile(values)
            percentiles[name] = {
                record["sample_id"]: float(value)
                for record, value in zip(records, ranked, strict=True)
            }
        for record in records:
            baseline_mean = float(np.mean([percentiles[m][record["sample_id"]] for m in METHODS]))
            record["baseline_mean_percentile"] = baseline_mean
            record["era_percentile"] = percentiles[ERA_NAME][record["sample_id"]]
            if record["label"]:
                negative = by_label_index[(0, record["pair_index"])]
                record["paired_negative_id"] = negative["sample_id"]
                record["era_pair_margin"] = record["U_ERA"] - negative["U_ERA"]
            candidates.append(record)

    positives = [record for record in candidates if record["label"]]
    negatives = [record for record in candidates if not record["label"]]
    success_pool = [record for record in positives if record["era_pair_margin"] > 0]
    success = max(success_pool or positives, key=lambda record: record["era_pair_margin"])
    normal = min(
        negatives,
        key=lambda record: (record["baseline_mean_percentile"] + record["era_percentile"]) / 2,
    )
    failure = min(positives, key=lambda record: record["era_pair_margin"])
    chosen = [
        ("baseline_miss_era_success", success),
        ("normal_low_uncertainty", normal),
        ("era_failure", failure),
    ]
    output = []
    for case_type, record in chosen:
        output.append(
            {
                "case_type": case_type,
                "model": record["model"],
                "dataset": record["dataset"],
                "sample_id": record["sample_id"],
                "paired_negative_id": record.get("paired_negative_id"),
                "question": record.get("question"),
                "references": record.get("references"),
                "vision": record.get("vision"),
                "reasoning": record.get("reasoning"),
                "answer": record.get("answer"),
                "judge_hallucination": record.get("hallucination"),
                "judge_rating": record.get("rating"),
                "hallucination_types": record.get("hallucination_types"),
                "ppl": record["scores"]["perplexity"],
                "se": record["scores"]["semantic_entropy"],
                "umpire": record["scores"]["umpire"],
                "era": record["U_ERA"],
                "baseline_mean_percentile": record["baseline_mean_percentile"],
                "era_percentile": record["era_percentile"],
                "era_pair_margin": record.get("era_pair_margin"),
            }
        )
    (out_dir / "representative_cases.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def _fmt(value: float, low: float | None = None, high: float | None = None) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    if low is None or high is None or not np.isfinite(low) or not np.isfinite(high):
        return f"{value:.3f}"
    return f"{value:.3f} ({low:.3f}, {high:.3f})"


def write_markdown(
    path: Path,
    performance: list[dict],
    improvement: list[dict],
    distributions: list[dict],
    consistency: list[dict],
    layers: list[dict],
    components: list[dict],
    correctness_strata: list[dict],
    cases: list[dict],
) -> None:
    """Write tables and raw case records without interpretive discussion."""
    lines = [
        "# 5 实验二：ERA 对低不确定性幻觉的识别与分析",
        "",
        "> 本文档由 `scripts/analysis/exp2_era_results.py` 从固定困难子集、正式 Judge 标签、三种基线分数与 ERA component 自动生成。仅记录统计结果，不包含结论性分析。",
        "",
        "## 5.2 ERA 幻觉检测性能",
        "",
        "### 5.2.1 ERA 与 PPL / SE / UMPIRE 对比",
        "",
        "| 模型 | 方法 | n | AUROC（95% CI） | AUPRC（95% CI） | PRR（95% CI） |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for method in ALL_METHODS:
            row = next(r for r in performance if r["model"] == model and r["method"] == method)
            cells = [
                _fmt(row[m], row[f"{m}_ci_low"], row[f"{m}_ci_high"])
                for m in METRIC_FNS
            ]
            lines.append(
                f"| {model} | {DISPLAY_NAME[method]} | {row['n']} | " + " | ".join(cells) + " |"
            )
    lines += [
        "",
        "### 5.2.2 相对最优基线的性能提升",
        "",
        "| 模型 | 指标 | 最优基线 | 基线 | ERA | 绝对提升（95% CI） | 相对提升 | bootstrap p | Holm p |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in improvement:
        lines.append(
            f"| {row['model']} | {row['metric'].upper()} | {DISPLAY_NAME[row['best_baseline']]} | "
            f"{row['baseline_value']:.3f} | {row['era_value']:.3f} | "
            f"{_fmt(row['absolute_improvement'], row['ci_low'], row['ci_high'])} | "
            f"{row['relative_improvement_pct']:.1f}% | {row['p_bootstrap']:.4f} | {row['p_holm']:.4f} |"
        )

    lines += [
        "",
        "## 5.3 ERA 分数的区分能力",
        "",
        "### 5.3.1 LUH 与非幻觉样本的 ERA 分布",
        "",
        "| 模型 | 分组 | n | 均值 | 标准差 | 中位数 | Q1 | Q3 | 最小值 | 最大值 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in distributions:
        lines.append(
            f"| {row['model']} | {row['group']} | {row['n']} | {row['mean']:.4f} | "
            f"{row['std']:.4f} | {row['median']:.4f} | {row['q25']:.4f} | "
            f"{row['q75']:.4f} | {row['min']:.4f} | {row['max']:.4f} |"
        )
    lines += [
        "",
        "### 5.3.2 跨模型一致性",
        "",
        "| 模型 | n | ERA AUROC | Cliff's delta | LUH 中位数 | 非幻觉中位数 | 中位数差 | 方向一致 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in consistency:
        lines.append(
            f"| {row['model']} | {row['n']} | {row['era_auroc']:.3f} | "
            f"{row['cliffs_delta']:.3f} | {row['luh_median']:.4f} | "
            f"{row['non_hallucination_median']:.4f} | {row['median_gap']:.4f} | "
            f"{'是' if row['direction_consistent'] else '否'} |"
        )

    lines += [
        "",
        "## 5.5 消融实验",
        "",
        "### 5.5.1 Decoder Layer 消融",
        "",
        "| 模型 | 层组合 | 层编号 | AUROC（95% CI） | AUPRC | PRR |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in layers:
        lines.append(
            f"| {row['model']} | {row['layer_set']} | {row['layers']} | "
            f"{_fmt(row['auroc'], row['auroc_ci_low'], row['auroc_ci_high'])} | "
            f"{row['auprc']:.3f} | {row['prr']:.3f} |"
        )
    lines += [
        "",
        "完整逐层结果见 `results/analysis/exp2/decoder_layer_sweep.csv`。",
        "",
        "### 5.5.2 Vision / Reasoning 归因分量消融",
        "",
        "| 模型 | 分量 | AUROC（95% CI） | AUPRC | PRR |",
        "|---|---|---:|---:|---:|",
    ]
    variant_names = {
        "vision_reasoning": "Vision + Reasoning",
        "vision_only": "Vision only",
        "reasoning_only": "Reasoning only",
    }
    for row in components:
        lines.append(
            f"| {row['model']} | {variant_names[row['variant']]} | "
            f"{_fmt(row['auroc'], row['auroc_ci_low'], row['auroc_ci_high'])} | "
            f"{row['auprc']:.3f} | {row['prr']:.3f} |"
        )

    lines += [
        "",
        "### 5.5.3 按答案正确性分层的 ERA 消融",
        "",
        "| 模型 | 正确性分层 | n | 幻觉 / 非幻觉 | AUROC（95% CI） | AUPRC（95% CI） | PRR（95% CI） |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in correctness_strata:
        cells = [
            _fmt(row[m], row[f"{m}_ci_low"], row[f"{m}_ci_high"])
            for m in METRIC_FNS
        ]
        lines.append(
            f"| {row['model']} | {'正确（C=1）' if row['correct'] else '错误（C=0）'} | "
            f"{row['n']} | {row['n_hallucination']} / {row['n_non_hallucination']} | "
            + " | ".join(cells)
            + " |"
        )

    case_titles = {
        "baseline_miss_era_success": "5.6.1 Baseline 漏检、ERA 成功案例",
        "normal_low_uncertainty": "5.6.2 正常低不确定性样本",
        "era_failure": "5.6.3 ERA 失败案例",
    }
    lines += ["", "## 5.6 典型案例分析", ""]
    for case in cases:
        lines += [
            f"### {case_titles[case['case_type']]}",
            "",
            "| 字段 | 内容 |",
            "|---|---|",
            f"| 模型 / 数据集 / 样本 | {case['model']} / {case['dataset']} / `{case['sample_id']}` |",
            f"| 配对负样本 | `{case.get('paired_negative_id') or 'N/A'}` |",
            f"| 问题 | {str(case.get('question') or '').replace('|', '\\|')} |",
            f"| 参考答案 | {str(case.get('references') or '').replace('|', '\\|')} |",
            f"| `<vision>` | {str(case.get('vision') or '').replace('|', '\\|')} |",
            f"| `<reasoning>` | {str(case.get('reasoning') or '').replace('|', '\\|')} |",
            f"| `<answer>` | {str(case.get('answer') or '').replace('|', '\\|')} |",
            f"| Judge | hallucination={case['judge_hallucination']}, rating={case['judge_rating']}, types={case['hallucination_types']} |",
            f"| PPL / SE / UMPIRE / ERA | {case['ppl']:.6f} / {case['se']:.6f} / {case['umpire']:.6f} / {case['era']:.6f} |",
            f"| 基线平均百分位 / ERA 百分位 | {case['baseline_mean_percentile']:.4f} / {case['era_percentile']:.4f} |",
            f"| ERA 配对差值 | {_fmt(case.get('era_pair_margin'))} |",
            "",
        ]

    lines += [
        "## 5.7 小结",
        "",
        "本节暂仅保留上述数值结果与原始案例记录，不加入结论性分析。",
        "",
        "### 输出文件",
        "",
        "- `era_vs_baselines.csv`",
        "- `era_improvement.csv`",
        "- `era_sample_scores.csv`",
        "- `era_distribution_summary.csv`",
        "- `cross_model_consistency.csv`",
        "- `decoder_layer_ablation.csv`",
        "- `decoder_layer_sweep.csv`",
        "- `component_ablation.csv`",
        "- `correctness_stratified_ablation.csv`",
        "- `representative_cases.json`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Experiment 2 ERA numerical results")
    parser.add_argument(
        "--subset",
        type=Path,
        default=RESULTS / "analysis" / "luh" / "per_model_subsets.json",
    )
    parser.add_argument(
        "--components-dir", type=Path, default=RESULTS / "era_components"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS / "analysis" / "exp2"
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=PROJECT_ROOT / "docs" / "实验二结果.md"
    )
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_model, _ = load_experiment_data(args.subset, args.components_dir, args.layers)
    performance, improvement = performance_tables(
        by_model, args.bootstrap_samples, args.output_dir
    )
    distributions, consistency = distribution_tables(by_model, args.output_dir)
    layers, components = ablation_tables(
        by_model, args.bootstrap_samples, args.output_dir
    )
    correctness_strata = correctness_stratified_ablation(
        by_model, args.bootstrap_samples, args.output_dir
    )
    cases = select_cases(by_model, args.output_dir)
    write_markdown(
        args.markdown_output,
        performance,
        improvement,
        distributions,
        consistency,
        layers,
        components,
        correctness_strata,
        cases,
    )
    manifest = {
        "subset": str(args.subset.resolve()),
        "components_dir": str(args.components_dir.resolve()),
        "layers": list(args.layers),
        "bootstrap_samples": args.bootstrap_samples,
        "models": {model: len(records) for model, records in by_model.items()},
        "markdown_output": str(args.markdown_output.resolve()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Experiment 2 numerical results: {args.output_dir}")
    print(f"Markdown: {args.markdown_output}")


if __name__ == "__main__":
    main()
