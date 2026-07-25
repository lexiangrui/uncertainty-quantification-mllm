from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Iterator

from .base import BenchmarkSample
from .hallusionbench import iter_hallusionbench
from .mmvet import iter_mmvet
from .vilp import iter_vilp


LOADERS = {
    "vilp": iter_vilp,
    "hallusionbench": iter_hallusionbench,
    "mmvet": iter_mmvet,
}


def iter_dataset(name: str, source: str | Path, limit: int | None = None) -> Iterator[BenchmarkSample]:
    if name not in LOADERS:
        raise ValueError(f"unknown dataset: {name}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    iterator = LOADERS[name](Path(source))
    yield from iterator if limit is None else islice(iterator, limit)
