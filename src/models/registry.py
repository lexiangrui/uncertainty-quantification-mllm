from __future__ import annotations

from pathlib import Path

from .base import GenerationBackend


def load_generation_backend(
    family: str,
    model_path: str | Path,
    *,
    adapter_path: str | Path | None = None,
    max_num_seqs: int = 8,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
) -> GenerationBackend:
    from .vllm_backend import VLLMMultimodalBackend

    return VLLMMultimodalBackend(
        family,
        Path(model_path),
        adapter_path=Path(adapter_path) if adapter_path else None,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )


def load_replay_backend(
    family: str,
    model_path: str | Path,
    *,
    adapter_path: str | Path | None = None,
    attn_implementation: str | None = None,
):
    from .huggingface import HuggingFaceReplayBackend

    return HuggingFaceReplayBackend(
        family,
        Path(model_path),
        attn_implementation=attn_implementation,
        adapter_path=Path(adapter_path) if adapter_path else None,
    )
