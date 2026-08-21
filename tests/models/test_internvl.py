from types import SimpleNamespace

from PIL import Image

from src.models.internvl import dynamic_image_tiles


def _config(**overrides):
    values = {
        "force_image_size": 448,
        "dynamic_image_size": True,
        "min_dynamic_patch": 1,
        "max_dynamic_patch": 12,
        "use_thumbnail": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dynamic_tiles_match_internvl_aspect_ratio_policy() -> None:
    assert len(dynamic_image_tiles(Image.new("RGB", (448, 448)), _config())) == 1
    assert len(dynamic_image_tiles(Image.new("RGB", (896, 448)), _config())) == 3
    assert len(dynamic_image_tiles(Image.new("RGB", (896, 896)), _config())) == 5


def test_dynamic_tiles_can_be_disabled() -> None:
    config = _config(dynamic_image_size=False)
    assert len(dynamic_image_tiles(Image.new("RGB", (1792, 448)), config)) == 1
