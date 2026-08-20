from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

from src.datasets.hallusionbench import iter_hallusionbench


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def _row(visual_input: str, *, image: bytes | None) -> dict:
    return {
        "category": "VD",
        "subcategory": "figure",
        "set_id": "1",
        "figure_id": "1",
        "question_id": visual_input,
        "question": "Is this red?",
        "gt_answer": "1",
        "gt_answer_details": "The image is red.",
        "visual_input": visual_input,
        "image": image,
    }


def test_hallusionbench_treats_visual_input_2_as_image(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame([_row("2", image=_png_bytes())]).to_parquet(data / "image-00000-of-00001.parquet")
    pd.DataFrame([_row("0", image=None)]).to_parquet(data / "non_image-00000-of-00001.parquet")

    samples = list(iter_hallusionbench(data))

    assert samples[0].image is not None
    assert samples[0].metadata["visual_input"] is True
    assert samples[1].image is None
    assert samples[1].metadata["visual_input"] is False
