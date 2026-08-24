#!/usr/bin/env python3
"""Experiment-two analysis for Early Rationale Attribution (ERA).

The analysis is deliberately restricted to the fixed 200-positive/200-negative
LUH subsets created without using ERA.  It produces machine-readable tables for:

* baseline and ERA AUROC/AUPRC/PRR with clustered bootstrap confidence intervals;
* paired metric gains of ERA over every baseline;
* the image/question/vision/reasoning attention decomposition behind U_ERA;
* within-matched-pair ranking accuracy;
* layer and relative-depth-band ablations;
* dataset/correctness strata and correlations with baseline or length signals.

Run the script once per judge/subset definition.  Keeping the runner generic
allows the primary Terra analysis and the Gemini sensitivity analysis to use
identical code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy import stats as sps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    auprc,
    auroc,
    cluster_bootstrap_indices,
    prr,
)
from src.improvement.era import EPS, layer_features  # noqa: E402
from src.utils import load_jsonl_records  # noqa: E402

MODELS = ("llava", "qwen", "internvl")
DATASETS = ("vilp", "hallusionbench", "mmvet")
BASELINES = ("perplexity", "semantic_entropy", "umpire")
METHODS = (*BASELINES, "U_ERA")
METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float | None]] = {
    "auroc": auroc,
    "auprc": auprc,
    "prr": prr,
}
DESTINATIONS = ("image", "prompt_text", "vision", "reasoning", "answer")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile_interval(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - confidence) * 50.0
    low, high = np.percentile(array, (tail, 100.0 - tail))
    return float(low), float(high)


def bootstrap_values(
    scores: np.ndarray,
    labels: np.ndarray,
    replicates: list[np.ndarray],
    metric: Callable[[np.ndarray, np.ndarray], float | None],
) -> np.ndarray:
    values = [metric(scores[index], labels[index]) for index in replicates]
    return np.asarray([value for value in values if value is not None], dtype=float)


def two_sided_bootstrap_p(values: Iterable[float]) -> float:
    """Two-sided sign probability for a bootstrap distribution of a difference."""
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return float("nan")
    lower = (np.count_nonzero(values <= 0.0) + 1.0) / (values.size + 1.0)
    upper = (np.count_nonzero(values >= 0.0) + 1.0) / (values.size + 1.0)
    return float(min(1.0, 2.0 * min(lower, upper)))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise adjustment, preserving input order."""
    result = [float("nan")] * len(p_values)
    valid = [(index, value) for index, value in enumerate(p_values) if math.isfinite(value)]
    valid.sort(key=lambda item: item[1])
    running = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(valid):
        running = max(running, (count - rank) * value)
        result[index] = min(1.0, running)
    return result


def attention_features(payload: dict, layers: Iterable[int]) -> dict[str, float]:
    """Average answer-row attention features over selected decoder layers.

    Each stored mass is first normalized by heads and answer-token count.  The
    canonical ERA denominator excludes answer-to-answer attention, preventing
    the score from being mechanically dominated by answer length.
    """
    available = {int(layer): masses for layer, masses in payload["layer_masses"].items()}
    selected = [layer for layer in layers if layer in available]
    if not selected:
        raise ValueError(f"selected layers are absent; available={sorted(available)}")
    scale = float(payload["n_heads"] * payload["section_tokens"]["answer"])
    if scale <= 0:
        raise ValueError("ERA normalization scale must be positive")
    per_layer = []
    for layer in selected:
        answer_row = available[layer][2]
        per_layer.append(np.asarray(answer_row, dtype=float) / scale)
    per_layer_array = np.stack(per_layer)
    means = np.mean(per_layer_array, axis=0)
    result = {f"attn_{name}": float(means[index]) for index, name in enumerate(DESTINATIONS)}
    result["attn_external"] = result["attn_image"] + result["attn_prompt_text"]
    result["attn_internal"] = result["attn_vision"] + result["attn_reasoning"]
    external_by_layer = per_layer_array[:, 0] + per_layer_array[:, 1]
    internal_by_layer = per_layer_array[:, 2] + per_layer_array[:, 3]
    result["U_ERA"] = float(np.mean(
        internal_by_layer / (external_by_layer + internal_by_layer + EPS)
    ))
    result["U_ERA_with_answer"] = float(np.mean(
        internal_by_layer
        / (external_by_layer + internal_by_layer + per_layer_array[:, 4] + EPS)
    ))
    result["n_visual_tokens"] = float(payload["n_visual_tokens"])
    for section in ("vision", "reasoning", "answer"):
        result[f"n_{section}_tokens"] = float(payload["section_tokens"][section])
    result["n_rationale_tokens"] = result["n_vision_tokens"] + result["n_reasoning_tokens"]
    return result


def load_records(
    *,
    results_root: Path,
    subset_path: Path,
    judge_dir: Path,
    components_dir: Path,
    layers: list[int],
) -> dict[str, dict]:
    subsets = json.loads(subset_path.read_text(encoding="utf-8"))
    loaded: dict[str, dict] = {}
    for model in MODELS:
        positive_ids = list(subsets[model]["positive_ids"])
        negative_ids = list(subsets[model]["negative_ids"])
        if len(positive_ids) != len(negative_ids):
            raise ValueError(f"{model}: positive/negative counts differ")
        pair_id = {
            sample_id: index
            for index, pair in enumerate(zip(positive_ids, negative_ids, strict=True))
            for sample_id in pair
        }
        positive_set = set(positive_ids)
        target_ids = positive_set | set(negative_ids)
        rows: dict[str, dict] = {}
        for dataset in DATASETS:
            for obj in load_jsonl_records(results_root / "generation" / model / "greedy" / f"{dataset}.jsonl"):
                if obj.get("record_type") != "sample":
                    continue
                sample = obj.get("sample", {})
                sid = sample.get("sample_id")
                if sid in target_ids:
                    greedy = obj.get("greedy", {})
                    rows[sid] = {
                        "sample_id": sid,
                        "model": model,
                        "dataset": dataset,
                        "group_id": str(sample.get("group_id") or sid),
                        "pair_id": pair_id[sid],
                        "subset_label": 1 if sid in positive_set else 0,
                        "answer_tokens": float((greedy.get("signals") or {}).get("token_count") or 0),
                    }
            for obj in load_jsonl_records(results_root / "uq" / model / f"{dataset}.jsonl"):
                if obj.get("record_type") != "sample":
                    continue
                sid = obj.get("sample", {}).get("sample_id")
                if sid not in rows:
                    continue
                uq = obj.get("uq", {})
                scores = {}
                for method in BASELINES:
                    entry = uq.get(method, {})
                    if entry.get("valid") is True and entry.get("score") is not None:
                        scores[method] = float(entry["score"])
                rows[sid]["scores"] = scores
            for obj in load_jsonl_records(judge_dir / model / f"{dataset}.jsonl"):
                if obj.get("record_type") != "sample":
                    continue
                sid = obj.get("sample", {}).get("sample_id")
                if sid not in rows:
                    continue
                judge = obj.get("judge", {})
                if judge.get("valid") is True:
                    rows[sid].update({
                        "hallucination": 1 if judge.get("hallucination") else 0,
                        "correct": 1 if judge.get("correct") else 0,
                        "rating": int(judge.get("rating")),
                        "hallucination_types": list(judge.get("hallucination_types") or []),
                    })
            component_path = components_dir / model / f"{dataset}.jsonl"
            for obj in load_jsonl_records(component_path):
                if obj.get("record_type") != "sample":
                    continue
                sid = obj.get("sample", {}).get("sample_id")
                payload = obj.get("era")
                if sid in rows and payload:
                    rows[sid]["era_payload"] = payload
                    rows[sid]["era"] = attention_features(payload, layers)

        required = {
            sid for sid, row in rows.items()
            if "scores" in row and len(row["scores"]) == len(BASELINES)
            and "hallucination" in row and "era" in row
        }
        missing = target_ids - required
        if missing:
            raise ValueError(f"{model}: incomplete inputs for {len(missing)} subset samples")
        ordered = positive_ids + negative_ids
        selected = [rows[sid] for sid in ordered]
        if any(row["hallucination"] != row["subset_label"] for row in selected):
            raise ValueError(f"{model}: subset labels disagree with selected judge")
        loaded[model] = {
            "rows": selected,
            "positive_ids": positive_ids,
            "negative_ids": negative_ids,
        }
    return loaded


def arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    labels = np.asarray([row["hallucination"] for row in rows], dtype=int)
    groups = np.asarray([f'{row["dataset"]}:{row["group_id"]}' for row in rows], dtype=object)
    scores = {
        method: np.asarray([
            row["era"]["U_ERA"] if method == "U_ERA" else row["scores"][method]
            for row in rows
        ], dtype=float)
        for method in METHODS
    }
    return labels, groups, scores


def main_metrics(
    data: dict[str, dict], *, n_bootstrap: int, seed: int
) -> tuple[list[dict], dict[str, dict[str, np.ndarray]], dict[str, list[np.ndarray]]]:
    output: list[dict] = []
    boot: dict[str, dict[str, np.ndarray]] = {}
    replicates_by_model: dict[str, list[np.ndarray]] = {}
    for model in MODELS:
        rows = data[model]["rows"]
        labels, groups, scores = arrays(rows)
        replicates = cluster_bootstrap_indices(groups.tolist(), n_bootstrap=n_bootstrap, seed=seed)
        replicates_by_model[model] = replicates
        boot[model] = {}
        for method in METHODS:
            for metric_name, metric_fn in METRICS.items():
                value = metric_fn(scores[method], labels)
                values = bootstrap_values(scores[method], labels, replicates, metric_fn)
                low, high = percentile_interval(values)
                boot[model][f"{method}:{metric_name}"] = values
                output.append({
                    "model": model,
                    "n": len(rows),
                    "n_pos": int(labels.sum()),
                    "method": method,
                    "metric": metric_name,
                    "value": value,
                    "ci_low": low,
                    "ci_high": high,
                })
    return output, boot, replicates_by_model


def paired_deltas(
    data: dict[str, dict],
    boot: dict[str, dict[str, np.ndarray]],
) -> list[dict]:
    output = []
    for model in MODELS:
        rows = data[model]["rows"]
        labels, _, scores = arrays(rows)
        for baseline in BASELINES:
            for metric_name, metric_fn in METRICS.items():
                delta = metric_fn(scores["U_ERA"], labels) - metric_fn(scores[baseline], labels)
                values = boot[model][f"U_ERA:{metric_name}"] - boot[model][f"{baseline}:{metric_name}"]
                low, high = percentile_interval(values)
                output.append({
                    "model": model,
                    "metric": metric_name,
                    "baseline": baseline,
                    "era_minus_baseline": delta,
                    "ci_low": low,
                    "ci_high": high,
                    "p_bootstrap": two_sided_bootstrap_p(values),
                })
    for metric_name in METRICS:
        indices = [index for index, row in enumerate(output) if row["metric"] == metric_name]
        adjusted = holm_adjust([output[index]["p_bootstrap"] for index in indices])
        for index, value in zip(indices, adjusted, strict=True):
            output[index]["p_holm"] = value
    return output


def macro_metrics(main_rows: list[dict], boot: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    lookup = {(row["model"], row["method"], row["metric"]): row for row in main_rows}
    output = []
    for method in METHODS:
        for metric_name in METRICS:
            value = float(np.mean([lookup[(model, method, metric_name)]["value"] for model in MODELS]))
            values = np.mean(np.stack([boot[model][f"{method}:{metric_name}"] for model in MODELS]), axis=0)
            low, high = percentile_interval(values)
            output.append({
                "method": method,
                "metric": metric_name,
                "macro_value": value,
                "ci_low": low,
                "ci_high": high,
            })
    return output


def mechanism_analysis(
    data: dict[str, dict], replicates_by_model: dict[str, list[np.ndarray]]
) -> tuple[list[dict], list[dict], list[dict]]:
    summary, differences, pair_rows = [], [], []
    feature_names = (
        "attn_image", "attn_prompt_text", "attn_vision", "attn_reasoning",
        "attn_external", "attn_internal", "U_ERA", "U_ERA_with_answer",
        "n_visual_tokens", "n_vision_tokens", "n_reasoning_tokens",
        "n_rationale_tokens", "n_answer_tokens",
    )
    for model in MODELS:
        rows = data[model]["rows"]
        labels = np.asarray([row["hallucination"] for row in rows], dtype=int)
        for feature in feature_names:
            values = np.asarray([row["era"][feature] for row in rows], dtype=float)
            for label, group in ((1, "luh"), (0, "matched_negative")):
                selected = values[labels == label]
                summary.append({
                    "model": model,
                    "group": group,
                    "feature": feature,
                    "n": len(selected),
                    "mean": float(np.mean(selected)),
                    "std": float(np.std(selected)),
                    "median": float(np.median(selected)),
                    "q1": float(np.quantile(selected, 0.25)),
                    "q3": float(np.quantile(selected, 0.75)),
                })
            point = float(np.mean(values[labels == 1]) - np.mean(values[labels == 0]))
            replicates = replicates_by_model[model]
            boot_diff = []
            for index in replicates:
                lab = labels[index]
                if np.any(lab == 1) and np.any(lab == 0):
                    val = values[index]
                    boot_diff.append(float(np.mean(val[lab == 1]) - np.mean(val[lab == 0])))
            low, high = percentile_interval(boot_diff)
            differences.append({
                "model": model,
                "feature": feature,
                "mean_difference_luh_minus_negative": point,
                "ci_low": low,
                "ci_high": high,
            })
        positives = {row["pair_id"]: row for row in rows if row["hallucination"] == 1}
        negatives = {row["pair_id"]: row for row in rows if row["hallucination"] == 0}
        for feature in ("U_ERA", "attn_internal", "attn_external"):
            deltas = np.asarray([
                positives[pair]["era"][feature] - negatives[pair]["era"][feature]
                for pair in sorted(positives)
            ])
            wins = float(np.mean(deltas > 0) + 0.5 * np.mean(deltas == 0))
            generator = np.random.default_rng(0)
            samples = [
                float(np.mean(deltas[index] > 0) + 0.5 * np.mean(deltas[index] == 0))
                for index in generator.integers(0, len(deltas), size=(2000, len(deltas)))
            ]
            low, high = percentile_interval(samples)
            pair_rows.append({
                "model": model,
                "feature": feature,
                "n_pairs": len(deltas),
                "pair_win_rate": wins,
                "ci_low": low,
                "ci_high": high,
                "median_pair_difference": float(np.median(deltas)),
            })
    return summary, differences, pair_rows


def layer_analysis(data: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    per_layer, bands_out = [], []
    bands = {
        "early_third": lambda rel: rel < 1 / 3,
        "middle_third": lambda rel: 1 / 3 <= rel < 2 / 3,
        "late_third": lambda rel: rel >= 2 / 3,
    }
    for model in MODELS:
        rows = data[model]["rows"]
        labels = np.asarray([row["hallucination"] for row in rows], dtype=int)
        layer_sets = [set(layer_features(row["era_payload"])) for row in rows]
        common_layers = sorted(set.intersection(*layer_sets))
        total = max(common_layers) + 1
        for layer in common_layers:
            scores = np.asarray([layer_features(row["era_payload"])[layer]["U_ERA"] for row in rows])
            per_layer.append({
                "model": model,
                "layer": layer,
                "relative_depth": layer / max(1, total - 1),
                "auroc": auroc(scores, labels),
            })
        for band, keep in bands.items():
            selected = [layer for layer in common_layers if keep(layer / max(1, total - 1))]
            scores = np.asarray([
                np.mean([layer_features(row["era_payload"])[layer]["U_ERA"] for layer in selected])
                for row in rows
            ])
            bands_out.append({
                "model": model,
                "band": band,
                "layers": ",".join(map(str, selected)),
                "auroc": auroc(scores, labels),
            })
    return per_layer, bands_out


def stratified_analysis(data: dict[str, dict]) -> list[dict]:
    output = []
    for model in MODELS:
        rows = data[model]["rows"]
        rationale_lengths = np.asarray([row["era"]["n_rationale_tokens"] for row in rows])
        boundaries = np.quantile(rationale_lengths, (0.25, 0.50, 0.75))
        length_bins = np.searchsorted(boundaries, rationale_lengths, side="right")
        strata = {
            "dataset": [(dataset, [row for row in rows if row["dataset"] == dataset]) for dataset in DATASETS],
            "correctness": [(str(value), [row for row in rows if row["correct"] == value]) for value in (0, 1)],
            "rationale_length_quartile": [
                (f"Q{value + 1}", [row for index, row in enumerate(rows) if length_bins[index] == value])
                for value in range(4)
            ],
        }
        for stratum_type, groups in strata.items():
            for stratum, selected in groups:
                if len(selected) < 4:
                    continue
                labels, _, scores = arrays(selected)
                if len(np.unique(labels)) < 2:
                    continue
                for method in METHODS:
                    output.append({
                        "model": model,
                        "stratum_type": stratum_type,
                        "stratum": stratum,
                        "n": len(selected),
                        "n_pos": int(labels.sum()),
                        "method": method,
                        "auroc": auroc(scores[method], labels),
                    })
    return output


def correlation_analysis(data: dict[str, dict]) -> list[dict]:
    output = []
    for model in MODELS:
        rows = data[model]["rows"]
        era = np.asarray([row["era"]["U_ERA"] for row in rows])
        comparators = {
            **{method: np.asarray([row["scores"][method] for row in rows]) for method in BASELINES},
            "answer_tokens": np.asarray([row["answer_tokens"] for row in rows]),
            "vision_tokens": np.asarray([row["era"]["n_vision_tokens"] for row in rows]),
            "reasoning_tokens": np.asarray([row["era"]["n_reasoning_tokens"] for row in rows]),
            "rationale_tokens": np.asarray([row["era"]["n_rationale_tokens"] for row in rows]),
            "visual_input_tokens": np.asarray([row["era"]["n_visual_tokens"] for row in rows]),
        }
        for name, values in comparators.items():
            if np.unique(era).size < 2 or np.unique(values).size < 2:
                statistic, p_value = float("nan"), float("nan")
            else:
                result = sps.spearmanr(era, values)
                statistic, p_value = float(result.statistic), float(result.pvalue)
            output.append({
                "model": model,
                "variable": name,
                "n": len(rows),
                "spearman": statistic,
                "p_value": p_value,
            })
    return output


def confounder_analysis(
    data: dict[str, dict], replicates_by_model: dict[str, list[np.ndarray]]
) -> list[dict]:
    """Compare ERA with length-only scores and a label-free residualized ERA score."""
    output = []
    for model in MODELS:
        rows = data[model]["rows"]
        labels = np.asarray([row["hallucination"] for row in rows], dtype=int)
        era = np.asarray([row["era"]["U_ERA"] for row in rows])
        vision = np.asarray([row["era"]["n_vision_tokens"] for row in rows])
        reasoning = np.asarray([row["era"]["n_reasoning_tokens"] for row in rows])
        answer = np.asarray([row["era"]["n_answer_tokens"] for row in rows])
        visual = np.asarray([row["era"]["n_visual_tokens"] for row in rows])
        design = np.column_stack([
            np.ones(len(rows)), np.log1p(vision), np.log1p(reasoning),
            np.log1p(answer), np.log1p(visual),
        ])
        fitted = design @ np.linalg.lstsq(design, era, rcond=None)[0]
        scores = {
            "U_ERA": era,
            "vision_tokens": vision,
            "reasoning_tokens": reasoning,
            "rationale_tokens": vision + reasoning,
            "answer_tokens": answer,
            "U_ERA_residualized_lengths": era - fitted,
        }
        for name, values in scores.items():
            point = auroc(values, labels)
            bootstrap = bootstrap_values(values, labels, replicates_by_model[model], auroc)
            low, high = percentile_interval(bootstrap)
            output.append({
                "model": model,
                "score": name,
                "auroc": point,
                "ci_low": low,
                "ci_high": high,
            })
    return output


def render_summary(
    *, judge_name: str, main_rows: list[dict], delta_rows: list[dict], macro_rows: list[dict]
) -> str:
    main = {(r["model"], r["method"], r["metric"]): r for r in main_rows}
    delta = {(r["model"], r["baseline"], r["metric"]): r for r in delta_rows}
    lines = [
        f"# Experiment 2 ERA analysis summary ({judge_name})",
        "",
        "## Main AUROC",
        "",
        "| Model | PPL | SE | UMPIRE | ERA (95% CI) | ERA - best baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        base = max(BASELINES, key=lambda method: main[(model, method, "auroc")]["value"])
        era = main[(model, "U_ERA", "auroc")]
        gain = delta[(model, base, "auroc")]
        lines.append(
            f"| {model} | {main[(model, 'perplexity', 'auroc')]['value']:.3f} | "
            f"{main[(model, 'semantic_entropy', 'auroc')]['value']:.3f} | "
            f"{main[(model, 'umpire', 'auroc')]['value']:.3f} | "
            f"{era['value']:.3f} [{era['ci_low']:.3f}, {era['ci_high']:.3f}] | "
            f"{gain['era_minus_baseline']:+.3f} [{gain['ci_low']:+.3f}, {gain['ci_high']:+.3f}] |"
        )
    lines.extend(["", "## Macro metrics", "", "| Method | Metric | Value | 95% CI |", "|---|---|---:|---:|"])
    for row in macro_rows:
        lines.append(
            f"| {row['method']} | {row['metric']} | {row['macro_value']:.3f} | "
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] |"
        )
    lines.extend([
        "",
        "The full CSV tables include paired gains, Holm-adjusted bootstrap sign probabilities,",
        "attention decomposition, matched-pair accuracy, layer ablations, strata, and correlations.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ERA on fixed LUH subsets")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--subset", type=Path)
    parser.add_argument("--judge-dir", type=Path)
    parser.add_argument("--components-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--judge-name", default="primary")
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.subset = args.subset or args.results_root / "analysis/luh/per_model_subsets.json"
    args.judge_dir = args.judge_dir or args.results_root / "judging_gpt_5_6_terra"
    args.components_dir = args.components_dir or args.results_root / "era_components"
    args.output_dir = args.output_dir or args.results_root / "analysis/era_results"
    return args


def main() -> None:
    args = parse_args()
    data = load_records(
        results_root=args.results_root,
        subset_path=args.subset,
        judge_dir=args.judge_dir,
        components_dir=args.components_dir,
        layers=args.layers,
    )
    main_rows, boot, replicates = main_metrics(
        data, n_bootstrap=args.bootstrap_samples, seed=args.seed
    )
    delta_rows = paired_deltas(data, boot)
    macro_rows = macro_metrics(main_rows, boot)
    mechanism, mechanism_diff, pair_rows = mechanism_analysis(data, replicates)
    layer_rows, band_rows = layer_analysis(data)
    strata_rows = stratified_analysis(data)
    correlation_rows = correlation_analysis(data)
    confounder_rows = confounder_analysis(data, replicates)

    out = args.output_dir
    write_csv(out / "d1_main_metrics.csv", main_rows)
    write_csv(out / "d2_paired_deltas.csv", delta_rows)
    write_csv(out / "d2_macro_metrics.csv", macro_rows)
    write_csv(out / "d3_mechanism_summary.csv", mechanism)
    write_csv(out / "d3_mechanism_differences.csv", mechanism_diff)
    write_csv(out / "d3_matched_pair_analysis.csv", pair_rows)
    write_csv(out / "d4_layer_sweep.csv", layer_rows)
    write_csv(out / "d4_layer_bands.csv", band_rows)
    write_csv(out / "d5_stratified_auroc.csv", strata_rows)
    write_csv(out / "d6_correlations.csv", correlation_rows)
    write_csv(out / "d6_confounder_checks.csv", confounder_rows)
    summary = render_summary(
        judge_name=args.judge_name,
        main_rows=main_rows,
        delta_rows=delta_rows,
        macro_rows=macro_rows,
    )
    (out / "report.md").write_text(summary, encoding="utf-8")
    manifest = {
        "judge_name": args.judge_name,
        "results_root": str(args.results_root),
        "subset": str(args.subset),
        "judge_dir": str(args.judge_dir),
        "components_dir": str(args.components_dir),
        "layers": args.layers,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "models": {model: len(data[model]["rows"]) for model in MODELS},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary)
    print(f"output_dir={out}")


if __name__ == "__main__":
    main()
