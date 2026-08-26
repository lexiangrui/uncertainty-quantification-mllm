#!/usr/bin/env python3
"""Export the three fixed Experiment 2 case images from official datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.hallusionbench import iter_hallusionbench  # noqa: E402
from src.datasets.vilp import iter_vilp  # noqa: E402


TARGETS = {
    "hallusionbench-image-VD-math-11-1-1": "exp2_case1_hallusionbench_math.jpg",
    "vilp-103-case1": "exp2_case2_vilp_broom.jpg",
    "vilp-17-case2": "exp2_case3_vilp_spider.jpg",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "report" / "figures",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    sources = (
        iter_hallusionbench(args.data_root / "HallusionBench" / "data"),
        iter_vilp(args.data_root / "vilp" / "ViLP.parquet"),
    )
    for samples in sources:
        for sample in samples:
            filename = TARGETS.get(sample.sample_id)
            if filename is None:
                continue
            if sample.image is None:
                raise ValueError(f"{sample.sample_id} has no image")
            sample.image.convert("RGB").save(
                args.output_dir / filename,
                format="JPEG",
                quality=95,
                subsampling=0,
                dpi=(300, 300),
            )
            found.add(sample.sample_id)

    missing = sorted(set(TARGETS) - found)
    if missing:
        raise ValueError(f"case images not found: {missing}")
    print(f"Exported {len(found)} case images to {args.output_dir}")


if __name__ == "__main__":
    main()
