"""DeBERTa entailment model — faithful to official code.

Uses ``microsoft/deberta-v2-xlarge-mnli`` fine-tuned on MNLI.
Output: 2 = entailment, 1 = neutral, 0 = contradiction.
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)


class DebertaEntailment:
    """DeBERTa-v2-xlarge-mnli for 3-way NLI entailment."""

    def __init__(self, model_id: str | None = None):
        model_id = model_id or os.environ.get(
            "DEBERTA_MNLI_MODEL", "microsoft/deberta-v2-xlarge-mnli"
        )
        logger.info("Loading DeBERTa entailment: %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id
        ).to(DEVICE)
        self.model.eval()
        logger.info("DeBERTa loaded on %s.", DEVICE)

    @torch.no_grad()
    def check_implication(self, text1: str, text2: str) -> int:
        """Return 2=entailment, 1=neutral, 0=contradiction."""
        inputs = self.tokenizer(
            text1, text2, return_tensors="pt", truncation=True
        ).to(DEVICE)
        outputs = self.model(**inputs)
        logits = outputs.logits
        # DeBERTa MNLI: 0=contradiction, 1=neutral, 2=entailment
        prediction = torch.argmax(F.softmax(logits, dim=1)).cpu().item()
        return prediction
