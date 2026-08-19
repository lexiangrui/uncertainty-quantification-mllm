from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image


def required_text(value: Any, field: str, sample_id: str) -> str:
    """Return non-empty scalar text and reject pandas-style missing values."""
    if value is None:
        raise ValueError(f"missing {field} for {sample_id}")
    try:
        import pandas as pd

        missing = pd.isna(value)
        if missing is pd.NA or (not hasattr(missing, "__len__") and bool(missing)):
            raise ValueError(f"missing {field} for {sample_id}")
    except ValueError:
        raise
    except (TypeError, AttributeError):
        pass
    text = str(value).strip()
    if not text:
        raise ValueError(f"empty {field} for {sample_id}")
    return text


def decode_image(value: Any, sample_id: str) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        with Image.open(BytesIO(value)) as image:
            return image.convert("RGB")
    if isinstance(value, dict) and isinstance(value.get("bytes"), bytes):
        with Image.open(BytesIO(value["bytes"])) as image:
            return image.convert("RGB")
    if isinstance(value, dict) and value.get("path"):
        with Image.open(value["path"]) as image:
            return image.convert("RGB")
    raise TypeError(f"unsupported image value for {sample_id}: {type(value).__name__}")


def require_columns(actual: list[str], required: set[str], source: str) -> None:
    missing = sorted(required - set(actual))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
