#!/usr/bin/env python3
"""Seed an ERA component directory from compatible existing component rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = ("vilp", "hallusionbench", "mmvet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample-ids-file", required=True, type=Path)
    args = parser.parse_args()

    values = [line.strip() for line in args.sample_ids_file.read_text().splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError("sample ID file contains duplicates")
    target = set(values)
    output_model_dir = args.output_dir / args.model
    output_model_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for dataset in DATASETS:
        source = args.source_dir / args.model / f"{dataset}.jsonl"
        output = output_model_dir / f"{dataset}.jsonl"
        with source.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows or rows[0].get("record_type") != "run":
            raise ValueError(f"missing ERA run header: {source}")
        header = rows[0]
        header["run"]["sample_filter"] = sorted(target)
        selected = [
            row
            for row in rows[1:]
            if row.get("sample", {}).get("sample_id") in target
        ]
        found = [row["sample"]["sample_id"] for row in selected]
        if len(found) != len(set(found)):
            raise ValueError(f"duplicate reusable sample IDs in {source}")
        with output.open("w", encoding="utf-8") as handle:
            for row in [header, *selected]:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        total += len(selected)
        print(f"{dataset}: seeded={len(selected)} -> {output}")
    print(f"total_seeded={total} target={len(target)} pending={len(target) - total}")


if __name__ == "__main__":
    main()
