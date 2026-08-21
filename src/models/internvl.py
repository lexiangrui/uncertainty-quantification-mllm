"""Shared helpers for the original InternVL checkpoint."""

from __future__ import annotations

from PIL import Image

INTERNVL_SYSTEM_PROMPT = (
    "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、"
    "清华大学及多家合作单位联合开发的多模态大模型。"
)


def _target_ratios(min_num: int, max_num: int) -> list[tuple[int, int]]:
    ratios = {
        (width, height)
        for count in range(min_num, max_num + 1)
        for width in range(1, count + 1)
        for height in range(1, count + 1)
        if min_num <= width * height <= max_num
    }
    return sorted(ratios, key=lambda ratio: ratio[0] * ratio[1])


def _closest_ratio(
    aspect_ratio: float,
    ratios: list[tuple[int, int]],
    *,
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best = (1, 1)
    best_diff = float("inf")
    area = width * height
    for ratio in ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff:
            best_diff = diff
            best = ratio
        elif diff == best_diff:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best = ratio
    return best


def dynamic_image_tiles(image: Image.Image, config) -> list[Image.Image]:
    """Apply the same dynamic tiling policy used by vLLM's InternVL processor."""
    image = image.convert("RGB")
    image_size = int(getattr(config, "force_image_size", None) or 448)
    dynamic = bool(getattr(config, "dynamic_image_size", False))
    min_num = int(getattr(config, "min_dynamic_patch", 1)) if dynamic else 1
    max_num = int(getattr(config, "max_dynamic_patch", 1)) if dynamic else 1
    use_thumbnail = bool(getattr(config, "use_thumbnail", False))

    width, height = image.size
    ratio = _closest_ratio(
        width / height,
        _target_ratios(min_num, max_num),
        width=width,
        height=height,
        image_size=image_size,
    )
    columns, rows = ratio
    resized = image.resize((image_size * columns, image_size * rows))
    tiles = []
    for index in range(columns * rows):
        left = (index % columns) * image_size
        top = (index // columns) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles
