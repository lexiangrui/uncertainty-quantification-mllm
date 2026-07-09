"""Backend registry."""

from __future__ import annotations

from .base import Backend
from .llava import LlavaBackend

BACKEND_MAP: dict[str, type[Backend]] = {
    "llava": LlavaBackend,
}


def build_backend(name: str, **kwargs) -> Backend:
    if name not in BACKEND_MAP:
        raise ValueError(f"Unknown backend {name!r}; choose from {sorted(BACKEND_MAP)}")
    return BACKEND_MAP[name](**kwargs)


__all__ = [
    "Backend",
    "LlavaBackend",
    "BACKEND_MAP",
    "build_backend",
]
