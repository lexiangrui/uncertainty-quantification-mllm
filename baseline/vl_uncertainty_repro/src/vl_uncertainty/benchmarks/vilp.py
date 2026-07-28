from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

from .base import Benchmark


class VILP(Benchmark):
    benchmark_type = "free_form"

    def __init__(self):
        root = Path(os.environ.get("VILP_PARQUET_PATH", "/opt/lexiangrui/datasets/vilp/ViLP.parquet"))
        self.ds = pd.read_parquet(root)

    def obtain_size(self) -> int:
        return len(self.ds) * 2

    def retrieve(self, idx: int) -> dict:
        original_idx, offset = divmod(idx, 2)
        case = offset + 1
        row = self.ds.iloc[original_idx]
        image = Image.open(BytesIO(row[f"image{case}"])).convert("RGB")
        return {
            "idx": idx,
            "img": image,
            "question": f"{row['question']}\nNOTE: Provide only the final answer. Do not provide unrelated details.",
            "gt_ans": str(row[f"answer{case}"]),
            "subset": f"case{case}",
        }
