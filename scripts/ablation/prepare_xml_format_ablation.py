#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ablation.xml_format import prepare_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the shared XML-format ablation sample manifest.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--production-generation-root",
        type=Path,
        default=PROJECT_ROOT / "results/generation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/ablation/xml_format/sample_manifest.json",
    )
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare_manifest(
        data_root=args.data_root,
        production_generation_root=args.production_generation_root,
        output=args.output,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(
        f"prepared {manifest['sample_size']} samples from {manifest['candidate_count']} candidates; "
        f"dataset_counts={manifest['dataset_counts']} output={args.output}"
    )


if __name__ == "__main__":
    main()
