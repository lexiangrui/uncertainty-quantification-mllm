"""Semantic Uncertainty reproduction package.

Reproduces "Detecting Hallucinations in Large Language Models Using
Semantic Entropy" (Farquhar et al., Nature 2024) for multimodal models.
"""

from sem_unc.models import MODEL_MAP, build_model
from sem_unc.datasets import DATASET_MAP, build_dataset
from sem_unc.entailment import build_entailment_model
from sem_unc.semantic_entropy import compute_semantic_entropy
from sem_unc.metrics import compute_metrics

__all__ = [
    "MODEL_MAP",
    "build_model",
    "DATASET_MAP",
    "build_dataset",
    "build_entailment_model",
    "compute_semantic_entropy",
    "compute_metrics",
]
