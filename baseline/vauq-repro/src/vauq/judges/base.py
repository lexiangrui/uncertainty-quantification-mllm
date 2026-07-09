"""Judge abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Judge(ABC):
    """Decide whether ``prediction`` matches ``gold`` for a benchmark sample.

    ``sample`` is the dict returned by ``Benchmark.retrieve`` (may carry
    ``choices`` or other fields the judge needs). Return ``True``/``False``;
    judges that cannot label a sample may return ``None``.
    """

    @abstractmethod
    def judge(self, prediction: str, gold: Any, sample: dict[str, Any] | None = None) -> bool | None:
        raise NotImplementedError
