"""Benchmark registry."""

from __future__ import annotations

from .base import Benchmark
from .cvbench import CVBench
from .mmvet import MMVet
from .vilp import VILP

BENCHMARK_MAP: dict[str, type[Benchmark]] = {
    "mmvet": MMVet,
    "cvbench": CVBench,
    "vilp": VILP,
}


def build_benchmark(name: str, **kwargs) -> Benchmark:
    if name not in BENCHMARK_MAP:
        raise ValueError(f"Unknown benchmark {name!r}; choose from {sorted(BENCHMARK_MAP)}")
    return BENCHMARK_MAP[name](**kwargs)


__all__ = [
    "Benchmark",
    "MMVet",
    "CVBench",
    "VILP",
    "BENCHMARK_MAP",
    "build_benchmark",
]
