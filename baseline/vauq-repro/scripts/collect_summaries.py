#!/usr/bin/env python3
"""Collect VAUQ summary JSON files into a compact Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Collect VAUQ .summary.json files.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="results/paper_grid_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(Path(args.results_dir).glob("*.summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {})
        rows.append(
            {
                "file": path.name,
                "n": data.get("n"),
                "accuracy": data.get("accuracy"),
                "vauq_auroc": _metric(metrics, "vauq", "auroc"),
                "vauq_aupr": _metric(metrics, "vauq", "aupr"),
                "entropy_auroc": _metric(metrics, "entropy", "auroc"),
                "is_auroc": _metric(metrics, "is_score", "auroc"),
            }
        )

    lines = [
        "| file | n | acc | VAUQ AUROC | VAUQ AUPR | entropy AUROC | IS AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {file} | {n} | {accuracy} | {vauq_auroc} | {vauq_aupr} | "
            "{entropy_auroc} | {is_auroc} |".format(
                file=row["file"],
                n=row["n"] if row["n"] is not None else "",
                accuracy=_fmt(row["accuracy"]),
                vauq_auroc=_fmt(row["vauq_auroc"]),
                vauq_aupr=_fmt(row["vauq_aupr"]),
                entropy_auroc=_fmt(row["entropy_auroc"]),
                is_auroc=_fmt(row["is_auroc"]),
            )
        )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


def _metric(metrics: dict, name: str, key: str):
    return metrics.get(name, {}).get(key)


def _fmt(value) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value) * 100:.2f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
