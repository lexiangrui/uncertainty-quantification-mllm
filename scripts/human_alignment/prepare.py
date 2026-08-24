#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.human_alignment import build_alignment_workspace  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the blind human-alignment queue.")
    parser.add_argument("--gpt-dir", type=Path, default=PROJECT_ROOT / "results/judging_gpt_5_6_terra")
    parser.add_argument("--gemini-dir", type=Path, default=PROJECT_ROOT / "results/judging_gemini_3_7_flash")
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT / "results/human_alignment")
    parser.add_argument("--skip-images", action="store_true", help="Do not copy dataset images (mainly for tests).")
    args = parser.parse_args()
    result = build_alignment_workspace(
        gpt_dir=args.gpt_dir,
        gemini_dir=args.gemini_dir,
        workspace=args.workspace,
        export_images=not args.skip_images,
    )
    print(json.dumps({"counts": result["counts"], "warnings": result["warnings"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
