"""Dataset registry and factory."""

from __future__ import annotations

from .base import Dataset
from .multimodal import CVBenchDataset, MMVetDataset, VILPDataset
from .text_qa import (
    NaturalQuestionsDataset,
    SQuADDataset,
    SVAMPDataset,
    TriviaQADataset,
)

DATASET_MAP: dict[str, type[Dataset]] = {
    "cvbench": CVBenchDataset,
    "mmvet": MMVetDataset,
    "vilp": VILPDataset,
    "trivia_qa": TriviaQADataset,
    "squad": SQuADDataset,
    "nq": NaturalQuestionsDataset,
    "svamp": SVAMPDataset,
}


def build_dataset(name: str, **kwargs) -> Dataset:
    """Instantiate a dataset by name.

    Parameters
    ----------
    name:
        Registry key: ``"cvbench"``, ``"mmvet"``, ``"vilp"``.
    **kwargs:
        Forwarded to the dataset constructor.
    """
    cls = DATASET_MAP.get(name)
    if cls is None:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_MAP)}"
        )
    return cls(**kwargs)
