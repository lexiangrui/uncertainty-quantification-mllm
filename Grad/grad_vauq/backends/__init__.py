"""Grad-VAUQ backend registry."""

from __future__ import annotations

from .llava import GradLlavaBackend

BACKEND_MAP = {
    "llava": GradLlavaBackend,
}


def build_backend(name: str, **kwargs):
    if name not in BACKEND_MAP:
        raise ValueError(f"Unknown backend {name!r}; choose from {sorted(BACKEND_MAP)}")
    return BACKEND_MAP[name](**kwargs)


__all__ = ["GradLlavaBackend", "BACKEND_MAP", "build_backend"]
