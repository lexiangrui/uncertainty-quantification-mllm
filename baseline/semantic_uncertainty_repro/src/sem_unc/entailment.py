"""Entailment models for semantic clustering.

Uses DeBERTa-v2-xlarge-mnli as the default local entailment model.
GPT and LLaMA-based entailment are reserved as optional extensions.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)


class EntailmentModel:
    """Base class for entailment checking."""

    def check_implication(self, text1: str, text2: str) -> int:
        """Return 2=entailment, 1=neutral, 0=contradiction."""
        raise NotImplementedError

    def save_cache(self) -> None:
        """Save any prediction cache (no-op by default)."""
        pass


# ------------------------------------------------------------------
# DeBERTa entailment
# ------------------------------------------------------------------
class EntailmentDeberta(EntailmentModel):
    """DeBERTa-v2-xlarge-mnli fine-tuned on MNLI.

    Parameters
    ----------
    model_id:
        HuggingFace repo id or local directory path.
    """

    def __init__(self, model_id: str = "microsoft/deberta-v2-xlarge-mnli"):
        # Support env-var override for offline cluster usage.
        model_id = os.environ.get("DEBERTA_MNLI_MODEL", model_id)
        logger.info("Loading DeBERTa entailment model: %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id
        ).to(DEVICE)
        self.model.eval()
        logger.info("DeBERTa entailment model loaded on %s.", DEVICE)

    @torch.no_grad()
    def check_implication(self, text1: str, text2: str) -> int:
        """Check if text1 semantically entails text2."""
        inputs = self.tokenizer(
            text1, text2, return_tensors="pt", truncation=True
        ).to(DEVICE)
        outputs = self.model(**inputs)
        logits = outputs.logits
        # DeBERTa MNLI: 0=contradiction, 1=neutral, 2=entailment
        prediction = torch.argmax(F.softmax(logits, dim=1)).cpu().item()
        return prediction


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------
_ENTAILMENT_REGISTRY = {
    "deberta": EntailmentDeberta,
}


def build_entailment_model(name: str = "deberta", **kwargs) -> EntailmentModel:
    """Instantiate an entailment model.

    Parameters
    ----------
    name:
        ``"deberta"`` (default).  GPT / LLaMA slots are reserved.
    **kwargs:
        Forwarded to the constructor.
    """
    cls = _ENTAILMENT_REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"Unknown entailment model '{name}'. "
            f"Available: {list(_ENTAILMENT_REGISTRY)}"
        )
    return cls(**kwargs)
