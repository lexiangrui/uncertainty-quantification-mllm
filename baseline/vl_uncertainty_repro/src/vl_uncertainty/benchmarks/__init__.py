"""Benchmark registry."""

from __future__ import annotations

from importlib import import_module

from .base import Benchmark

BENCHMARK_MAP: dict[str, str] = {
    "cvbench": "vl_uncertainty.benchmarks.cvbench:CVBench",
    "mmvet": "vl_uncertainty.benchmarks.mmvet:MMVet",
    "vilp": "vl_uncertainty.benchmarks.vilp:VILP",
}


def build_benchmark(name: str, **kwargs) -> Benchmark:
    if name not in BENCHMARK_MAP:
        raise ValueError(f"Unknown benchmark {name!r}; choose from {sorted(BENCHMARK_MAP)}")
    return _load_class(BENCHMARK_MAP[name])(**kwargs)


def _load_class(path: str):
    module_name, class_name = path.split(":", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def __getattr__(name: str):
    class_to_key = {
        "MMVet": "mmvet",
        "CVBench": "cvbench",
        "VILP": "vilp",
    }
    if name in class_to_key:
        return _load_class(BENCHMARK_MAP[class_to_key[name]])
    raise AttributeError(name)


__all__ = [
    "Benchmark",
    "MMVet",
    "CVBench",
    "VILP",
    "BENCHMARK_MAP",
    "build_benchmark",
]
