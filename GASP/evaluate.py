import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from config import SEED

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def interval(labels: np.ndarray, scores: np.ndarray, metric, repeats: int = 1000) -> list[float]:
    rng = np.random.default_rng(SEED)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    values = []
    for _ in range(repeats):
        indices = np.concatenate([rng.choice(positive, len(positive)), rng.choice(negative, len(negative))])
        values.append(metric(labels[indices], scores[indices]))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--judged-output", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    unresolved = [record for record in records if record.get("correct") is None]
    if unresolved:
        if args.judged_output is None:
            raise ValueError("--judged-output is required when records need Qwen judging")
        from config import JUDGE_MODEL
        from judge import QwenLLMJudge

        judge = QwenLLMJudge(JUDGE_MODEL)
        for record in unresolved:
            correct = judge.judge(record["question"], record["references"], record["prediction"])
            record["correct"] = correct
            record["error_label"] = int(not correct)
            record["judge"] = judge.name
            record["judge_result"] = judge.last_result
        args.judged_output.parent.mkdir(parents=True, exist_ok=True)
        with args.judged_output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    labels = np.asarray([record["error_label"] for record in records], dtype=np.int64)
    scores = np.asarray([record["scores"]["uncertainty"] for record in records], dtype=np.float64)
    metrics = {
        "samples": len(records),
        "accuracy": float(1.0 - labels.mean()),
        "sample_error_rate": float(labels.mean()),
        "mean_uncertainty": float(scores.mean()),
        "mean_runtime_seconds": float(np.mean([record["runtime_seconds"] for record in records])),
        "max_peak_memory_gb": float(max(record["peak_memory_gb"] for record in records)),
    }
    if set(labels.tolist()) == {0, 1}:
        metrics.update(
            auroc=float(roc_auc_score(labels, scores)),
            auroc_ci95=interval(labels, scores, roc_auc_score),
            auprc=float(average_precision_score(labels, scores)),
            auprc_ci95=interval(labels, scores, average_precision_score),
        )
        component_values = {
            "nll0": np.asarray(
                [record["scores"]["nll0"] for record in records], dtype=np.float64
            ),
            "predictive_uncertainty": np.asarray(
                [record["scores"]["predictive_uncertainty"] for record in records],
                dtype=np.float64,
            ),
            "visual_nll_dependency": np.asarray(
                [record["scores"]["visual_nll_dependency"] for record in records],
                dtype=np.float64,
            ),
            "visual_volume_dependency": np.asarray(
                [record["scores"]["visual_volume_dependency"] for record in records],
                dtype=np.float64,
            ),
            "visual_dependency": np.asarray(
                [record["scores"]["visual_dependency"] for record in records],
                dtype=np.float64,
            ),
            "visual_ungrounded_risk": np.asarray(
                [record["scores"]["visual_ungrounded_risk"] for record in records],
                dtype=np.float64,
            ),
            "uncertainty": scores,
        }
        predictive = component_values["predictive_uncertainty"]
        nll_only_uncertainty = 1.0 - (1.0 - predictive) * component_values[
            "visual_nll_dependency"
        ]
        volume_only_uncertainty = 1.0 - (1.0 - predictive) * component_values[
            "visual_volume_dependency"
        ]
        component_values.update(
            nll_only_uncertainty=nll_only_uncertainty,
            volume_only_uncertainty=volume_only_uncertainty,
        )
        correct_labels = 1 - labels
        metrics["component_ranking"] = {
            "nll0_error_ranking": {
                "mean": float(component_values["nll0"].mean()),
                "positive_class": "error",
                "auroc": float(roc_auc_score(labels, component_values["nll0"])),
                "auprc": float(average_precision_score(labels, component_values["nll0"])),
            },
            **{
                f"{name}_correctness_ranking": {
                    "mean": float(values.mean()),
                    "positive_class": "correct",
                    "auroc": float(roc_auc_score(correct_labels, values)),
                    "auprc": float(average_precision_score(correct_labels, values)),
                }
                for name, values in component_values.items()
                if name.endswith("dependency")
            },
            **{
                f"{name}_error_ranking": {
                    "mean": float(values.mean()),
                    "positive_class": "error",
                    "auroc": float(roc_auc_score(labels, values)),
                    "auprc": float(average_precision_score(labels, values)),
                }
                for name, values in component_values.items()
                if name in {"predictive_uncertainty", "visual_ungrounded_risk"}
            },
            "combined_uncertainty_error_ranking": {
                "mean": float(scores.mean()),
                "positive_class": "error",
                "auroc": float(roc_auc_score(labels, scores)),
                "auprc": float(average_precision_score(labels, scores)),
            },
            "nll_only_uncertainty_error_ranking": {
                "mean": float(nll_only_uncertainty.mean()),
                "positive_class": "error",
                "auroc": float(roc_auc_score(labels, nll_only_uncertainty)),
                "auroc_ci95": interval(labels, nll_only_uncertainty, roc_auc_score),
                "auprc": float(average_precision_score(labels, nll_only_uncertainty)),
            },
            "volume_only_uncertainty_error_ranking": {
                "mean": float(volume_only_uncertainty.mean()),
                "positive_class": "error",
                "auroc": float(roc_auc_score(labels, volume_only_uncertainty)),
                "auroc_ci95": interval(labels, volume_only_uncertainty, roc_auc_score),
                "auprc": float(average_precision_score(labels, volume_only_uncertainty)),
            },
        }
        for quantile, label in ((0.25, "highest_confidence_25pct"), (0.50, "highest_confidence_50pct")):
            threshold = float(np.quantile(predictive, quantile))
            subset = predictive <= threshold
            subset_labels = labels[subset]
            entry = {
                "selection": f"predictive_uncertainty <= {quantile:.2f} quantile",
                "threshold": threshold,
                "samples": int(subset.sum()),
                "errors": int(subset_labels.sum()),
                "sample_error_rate": float(subset_labels.mean()),
            }
            if set(subset_labels.tolist()) == {0, 1}:
                entry.update(
                    visual_ungrounded_risk_auroc=float(
                        roc_auc_score(subset_labels, component_values["visual_ungrounded_risk"][subset])
                    ),
                    combined_uncertainty_auroc=float(roc_auc_score(subset_labels, scores[subset])),
                    nll_only_uncertainty_auroc=float(
                        roc_auc_score(subset_labels, nll_only_uncertainty[subset])
                    ),
                    volume_only_uncertainty_auroc=float(
                        roc_auc_score(subset_labels, volume_only_uncertainty[subset])
                    ),
                )
            metrics.setdefault("confidence_strata", {})[label] = entry
    else:
        metrics["ranking_metrics_unavailable"] = "requires both correct and incorrect samples"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
