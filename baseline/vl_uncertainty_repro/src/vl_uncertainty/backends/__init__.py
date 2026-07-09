"""Backend registry."""

from __future__ import annotations

from importlib import import_module

from .base import Backend

BACKEND_MAP: dict[str, str] = {
    "llava": "vl_uncertainty.backends.llava:LlavaBackend",
    "huggingface_llm": "vl_uncertainty.backends.huggingface_llm:HuggingFaceLLMBackend",
}


def build_backend(name: str, **kwargs) -> Backend:
    if name not in BACKEND_MAP:
        raise ValueError(f"Unknown backend {name!r}; choose from {sorted(BACKEND_MAP)}")
    return _load_class(BACKEND_MAP[name])(**kwargs)


def _load_class(path: str):
    module_name, class_name = path.split(":", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def __getattr__(name: str):
    mapping = {"LlavaBackend": "llava", "HuggingFaceLLMBackend": "huggingface_llm"}
    if name in mapping:
        return _load_class(BACKEND_MAP[mapping[name]])
    raise AttributeError(name)


__all__ = ["Backend", "LlavaBackend", "HuggingFaceLLMBackend", "BACKEND_MAP", "build_backend"]
