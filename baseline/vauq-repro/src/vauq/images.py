from __future__ import annotations

from typing import Iterable


def ensure_rgb(image):
    if image is None:
        return None
    if hasattr(image, "convert"):
        return image.convert("RGB")
    raise TypeError(f"Expected a PIL image-like object, got {type(image)!r}")


def blank_like(image, fill: tuple[int, int, int] = (127, 127, 127)):
    from PIL import Image

    image = ensure_rgb(image)
    return Image.new("RGB", image.size, fill)


def mask_grid_patches(
    image,
    grid_hw: tuple[int, int],
    patch_indices: Iterable[int],
    fill: tuple[int, int, int] = (127, 127, 127),
):
    from PIL import ImageDraw

    image = ensure_rgb(image).copy()
    width, height = image.size
    gh, gw = grid_hw
    if gh <= 0 or gw <= 0:
        return image
    draw = ImageDraw.Draw(image)
    for idx in set(int(i) for i in patch_indices):
        row, col = divmod(idx, gw)
        if row >= gh:
            continue
        x0 = int(round(col * width / gw))
        x1 = int(round((col + 1) * width / gw))
        y0 = int(round(row * height / gh))
        y1 = int(round((row + 1) * height / gh))
        draw.rectangle([x0, y0, x1, y1], fill=fill)
    return image
