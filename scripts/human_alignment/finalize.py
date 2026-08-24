#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.human_alignment import finalize_aligned_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create official labels after all disagreements are adjudicated.")
    parser.add_argument("--gpt-dir", type=Path, default=PROJECT_ROOT / "results/judging_gpt_5_6_terra")
    parser.add_argument("--gemini-dir", type=Path, default=PROJECT_ROOT / "results/judging_gemini_3_7_flash")
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT / "results/human_alignment")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/judging")
    parser.add_argument("--human-adjudicator", default="lexiangrui")
    parser.add_argument(
        "--trusted-gpt-models",
        default="internvl",
        help="comma-separated model families whose disagreement fields follow GPT",
    )
    args = parser.parse_args()
    trusted_gpt_models = tuple(
        item.strip() for item in args.trusted_gpt_models.split(",") if item.strip()
    )
    result = finalize_aligned_results(
        gpt_dir=args.gpt_dir,
        gemini_dir=args.gemini_dir,
        workspace=args.workspace,
        output_dir=args.output,
        human_adjudicator=args.human_adjudicator,
        trusted_gpt_models=trusted_gpt_models,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
