from __future__ import annotations

from pathlib import Path

from .base import GenerationBackend
from .huggingface import HuggingFaceMultimodalBackend
from .vllm_backend import VLLMMultimodalBackend


def load_backend(
    family: str,
    model_path: str | Path,
    *,
    attn_implementation: str | None = None,
    adapter_path: str | Path | None = None,
    engine: str = "huggingface",
    max_num_seqs: int = 5,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
) -> GenerationBackend:
    if engine == "vllm":
        return VLLMMultimodalBackend(
            family,
            Path(model_path),
            adapter_path=Path(adapter_path) if adapter_path else None,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
    if engine != "huggingface":
        raise ValueError(f"unknown generation engine: {engine}")
    return HuggingFaceMultimodalBackend(
        family,
        Path(model_path),
        attn_implementation=attn_implementation,
        adapter_path=Path(adapter_path) if adapter_path else None,
    )
