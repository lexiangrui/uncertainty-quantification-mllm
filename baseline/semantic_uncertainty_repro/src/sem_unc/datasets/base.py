"""Abstract base class for datasets / benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Dataset(ABC):
    """Interface for QA / VQA datasets.

    Mirrors vauq-repro's ``Benchmark`` so both baselines can share
    the same dataset implementations.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Return one sample as a dict.

        Expected keys
        -------------
        id : str
            Unique sample identifier.
        question : str
            Question text (with any appended instructions).
        img : PIL.Image.Image
            Input image.
        gt_ans : str
            Ground-truth answer.
        choices : list[str] | None
            Multiple-choice options (optional).
        subset : str | None
            Subset tag for reporting (optional).
        """
        ...
