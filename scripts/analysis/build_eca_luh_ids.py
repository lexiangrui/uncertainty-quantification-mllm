#!/usr/bin/env python3
"""Build the ECA sweep-set sample IDs: the 400-sample LUH subsets.

Per project decision (2026-08-18) the layer sweep runs directly on the LUH
subsets (200 low-uncertainty hallucinations + 200 matched controls) to
maximize performance there.  NOTE: any band chosen this way is selected on
the evaluation set itself — reported numbers are in-sample for the band
choice and must be labelled as such.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_jsonl_records

MODELS = ("llava", "qwen", "internvl")
DATASETS = ("hallusionbench", "vilp", "mmvet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=Path, default=PROJECT_ROOT / "results/analysis/luh/per_model_subsets.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/analysis/eca/luh_ids")
    args = parser.parse_args()

    subsets = json.loads(args.subset.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        sub = subsets[model]
        ids = set(sub["positive_ids"]) | set(sub["negative_ids"])
        out = args.output_dir / f"{model}.txt"
        out.write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
        print(f"{model}: sweep_ids={len(ids)} -> {out}")


if __name__ == "__main__":
    main()
