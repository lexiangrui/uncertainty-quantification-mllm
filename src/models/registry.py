from __future__ import annotations

from pathlib import Path

from .base import GenerationBackend


def load_backend(
    family: str,
    model_path: str | Path,
    *,
    attn_implementation: str | None = None,
    adapter_path: str | Path | None = None,
) -> GenerationBackend:
    from .huggingface import HuggingFaceMultimodalBackend

    return HuggingFaceMultimodalBackend(
        family,
        Path(model_path),
        attn_implementation=attn_implementation,
        adapter_path=Path(adapter_path) if adapter_path else None,
    )
