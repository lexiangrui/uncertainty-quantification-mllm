from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from ._image import decode_image, require_columns
from .base import BenchmarkSample


def iter_vilp(path: Path) -> Iterator[BenchmarkSample]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"question"} | {f"image{i}" for i in range(1, 4)} | {
        f"answer{i}" for i in range(1, 4)
    }
    require_columns(list(frame.columns), required, str(path))
    for row_number, row in frame.iterrows():
        group_id = f"vilp-{row_number}"
        for case in range(1, 4):
            sample_id = f"{group_id}-case{case}"
            yield BenchmarkSample(
                sample_id=sample_id,
                group_id=group_id,
                dataset="vilp",
                split="test",
                question=str(row["question"]).strip(),
                references=(str(row[f"answer{case}"]).strip(),),
                image=decode_image(row[f"image{case}"], sample_id),
                metadata={"row_number": int(row_number), "case": case},
            )
