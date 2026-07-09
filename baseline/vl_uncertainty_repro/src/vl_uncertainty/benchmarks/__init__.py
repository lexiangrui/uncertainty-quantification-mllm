"""Benchmark registry."""

from __future__ import annotations

from importlib import import_module

from .base import Benchmark

BENCHMARK_MAP: dict[str, str] = {
    "mmvet": "vl_uncertainty.benchmarks.mmvet:MMVet",
    "llavabench": "vl_uncertainty.benchmarks.llavabench:LLaVABench",
    "mmmu": "vl_uncertainty.benchmarks.mmmu:MMMU",
    "scienceqa": "vl_uncertainty.benchmarks.scienceqa:ScienceQA",
    "trivia_qa": "vl_uncertainty.benchmarks.text_qa:TriviaQABenchmark",
}

DEFAULT_JUDGE: dict[str, str] = {
    "mmvet": "llm",
    "llavabench": "llm",
    "mmmu": "choice",
    "scienceqa": "choice",
    "trivia_qa": "none",
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
        "LLaVABench": "llavabench",
        "MMMU": "mmmu",
        "ScienceQA": "scienceqa",
    }
    if name in class_to_key:
        return _load_class(BENCHMARK_MAP[class_to_key[name]])
    raise AttributeError(name)


__all__ = [
    "Benchmark",
    "MMVet",
    "LLaVABench",
    "MMMU",
    "ScienceQA",
    "BENCHMARK_MAP",
    "DEFAULT_JUDGE",
    "build_benchmark",
]
