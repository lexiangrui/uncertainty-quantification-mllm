"""Visual-token adapters."""

from .base import VisionTokenAdapter
from .llava import LlavaVisualTokenAdapter

ADAPTER_MAP = {
    "llava": LlavaVisualTokenAdapter,
}


def build_adapter(name: str) -> VisionTokenAdapter:
    if name == "auto":
        name = "llava"
    if name not in ADAPTER_MAP:
        raise ValueError(f"Unknown adapter {name!r}; choose from {sorted(ADAPTER_MAP)}")
    return ADAPTER_MAP[name]()


__all__ = ["VisionTokenAdapter", "LlavaVisualTokenAdapter", "ADAPTER_MAP", "build_adapter"]
