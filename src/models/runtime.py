"""GPU-memory-aware production batch sizing."""
from __future__ import annotations

import torch


_DEFAULT_VLLM_MAX_MODEL_LEN = 4096
_VLLM_MAX_MODEL_LEN_BY_FAMILY = {
    # Audited production bound with the checked-in XML prompt and datasets:
    # 16,384 visual + 155 rendered text - 1 placeholder + 512 output = 17,050.
    "qwen2_5_vl": 18_000,
}


def visible_gpu_memory_gib() -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("hybrid generation requires a CUDA GPU")
    return torch.cuda.get_device_properties(0).total_memory / 2**30


def replay_batch_size(memory_gib: float) -> int:
    """Return the largest conservative HF replay batch for visible VRAM.

    The 32-GiB tier deliberately selects five: that size was validated on the
    target RTX 5090 nodes.  Runtime OOM splitting remains the final guard for
    unusually long multimodal sequences.
    """
    if memory_gib < 20:
        return 1
    if memory_gib < 24:
        return 2
    if memory_gib < 30:
        return 4
    if memory_gib < 48:
        return 5
    return 8


def vllm_max_num_seqs(memory_gib: float) -> int:
    if memory_gib < 20:
        return 2
    if memory_gib < 24:
        return 4
    if memory_gib < 30:
        return 6
    if memory_gib < 48:
        return 8
    return 16


def vllm_max_model_len(family: str) -> int:
    """Return the audited production context capacity for a model family."""
    return _VLLM_MAX_MODEL_LEN_BY_FAMILY.get(
        family, _DEFAULT_VLLM_MAX_MODEL_LEN
    )
