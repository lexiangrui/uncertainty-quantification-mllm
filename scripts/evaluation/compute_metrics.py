#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import run_metrics


def _formatted(entry: dict) -> str:
    if entry["value"] is None:
        return "N/A"
    value = f"{entry['value']:.4f}"
    if entry["ci_low"] is None:
        return f"{value} [no defined replicates]"
    return f"{value} [{entry['ci_low']:.4f}, {entry['ci_high']:.4f}]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge deferred UQ scores with judge labels for one model x dataset "
            "run and compute AUROC/AUPRC/PRR/ECE with cluster bootstrap CIs "
            "for the error and hallucination targets."
        )
    )
    parser.add_argument("--uq-input", required=True, type=Path)
    parser.add_argument("--judge-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--ece-bins", type=int, default=15)
    args = parser.parse_args()

    report = run_metrics(
        uq_input=args.uq_input,
        judge_input=args.judge_input,
        output=args.output,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        ece_bins=args.ece_bins,
    )

    counts = report["counts"]
    excluded = counts["excluded"]
    print(
        f"model_family={report['model_family']} dataset={report['dataset']} "
        f"evaluated={counts['evaluated']} clusters={counts['clusters']}"
    )
    print(
        f"excluded: invalid_judge={excluded['invalid_judge']} "
        f"missing_uq_record={excluded['missing_uq_record']} "
        f"invalid_uq_score={excluded['invalid_uq_score']} "
        f"uq_without_judge_record={excluded['uq_without_judge_record']}"
    )
    labels = report["labels"]
    print(
        f"accuracy={_formatted(labels['accuracy'])} "
        f"hallucination_rate={_formatted(labels['hallucination_rate'])}"
    )
    for target_name, target in report["targets"].items():
        print(
            f"\ntarget={target_name} positives={target['positives']} "
            f"negatives={target['negatives']} "
            f"positive_rate={target['positive_rate']:.4f}"
        )
        width = max(len(name) for name in report["uq_methods"])
        for name, metrics in target["methods"].items():
            print(
                f"  {name:<{width}}  "
                f"AUROC={_formatted(metrics['auroc'])}  "
                f"AUPRC={_formatted(metrics['auprc'])}  "
                f"PRR={_formatted(metrics['prr'])}  "
                f"ECE={_formatted(metrics['ece'])}"
            )
    print(f"\nreport written to {args.output}")


if __name__ == "__main__":
    main()
