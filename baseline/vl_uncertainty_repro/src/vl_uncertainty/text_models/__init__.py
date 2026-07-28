"""Text model registry."""

from __future__ import annotations

from importlib import import_module

from .base import TextModel

TEXT_MODEL_MAP: dict[str, str] = {
    "qwen": "vl_uncertainty.text_models.qwen:QwenTextModel",
}


def build_text_model(name: str, **kwargs) -> TextModel:
    if name not in TEXT_MODEL_MAP:
        raise ValueError(f"Unknown text model {name!r}; choose from {sorted(TEXT_MODEL_MAP)}")
    return _load_class(TEXT_MODEL_MAP[name])(**kwargs)


def _load_class(path: str):
    module_name, class_name = path.split(":", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def __getattr__(name: str):
    class_to_key = {
        "QwenTextModel": "qwen",
    }
    if name in class_to_key:
        return _load_class(TEXT_MODEL_MAP[class_to_key[name]])
    raise AttributeError(name)


__all__ = [
    "TextModel",
    "QwenTextModel",
    "TEXT_MODEL_MAP",
    "build_text_model",
]
