from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

from src.datasets.vilp import iter_vilp
import pytest


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_vilp_expands_three_cases(tmp_path: Path) -> None:
    image = _png_bytes()
    path = tmp_path / "ViLP.parquet"
    pd.DataFrame(
        [{"question": "color?", **{f"image{i}": image for i in range(1, 4)}, **{f"answer{i}": str(i) for i in range(1, 4)}}]
    ).to_parquet(path)
    samples = list(iter_vilp(path))
    assert [sample.sample_id for sample in samples] == [
        "vilp-0-case1",
        "vilp-0-case2",
        "vilp-0-case3",
    ]
    assert len({sample.group_id for sample in samples}) == 1


def test_vilp_rejects_missing_text_instead_of_stringifying_nan(tmp_path: Path) -> None:
    path = tmp_path / "ViLP.parquet"
    image = _png_bytes()
    pd.DataFrame(
        [{"question": float("nan"), **{f"image{i}": image for i in range(1, 4)}, **{f"answer{i}": str(i) for i in range(1, 4)}}]
    ).to_parquet(path)
    with pytest.raises(ValueError, match="missing question"):
        list(iter_vilp(path))
