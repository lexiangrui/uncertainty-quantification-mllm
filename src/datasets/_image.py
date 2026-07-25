from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image


def decode_image(value: Any, sample_id: str) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        return Image.open(BytesIO(value)).convert("RGB")
    if isinstance(value, dict) and isinstance(value.get("bytes"), bytes):
        return Image.open(BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, dict) and value.get("path"):
        return Image.open(value["path"]).convert("RGB")
    raise TypeError(f"unsupported image value for {sample_id}: {type(value).__name__}")


def require_columns(actual: list[str], required: set[str], source: str) -> None:
    missing = sorted(required - set(actual))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
