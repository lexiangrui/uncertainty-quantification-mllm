from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.datasets.mmvet import iter_mmvet


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def test_mmvet_rejects_missing_answer_instead_of_stringifying_nan(tmp_path: Path) -> None:
    path = tmp_path / "mmvet.parquet"
    pd.DataFrame(
        [{"question_id": "1", "question": "What?", "answer": float("nan"), "image": _png_bytes()}]
    ).to_parquet(path)
    with pytest.raises(ValueError, match="missing answer"):
        list(iter_mmvet(path))
