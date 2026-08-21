"""GPU-memory-aware production batch sizing."""
from __future__ import annotations

import torch


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
