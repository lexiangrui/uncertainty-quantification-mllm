"""Pairwise natural-language-inference judge used by Semantic Entropy."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class PairwiseNLIJudge(Protocol):
    model_id: str

    def check_pairs(self, pairs: list[tuple[str, str]]) -> list[bool]: ...


class NLIJudge:
    """Local DeBERTa NLI judge for premise/hypothesis pairs.

    The public ``judge`` method handles one pair; ``check_pairs`` is the
    batched interface consumed by semantic-equivalence clustering.
    """

    name = "nli"

    def __init__(
        self, model_path: Path, *, batch_size: int = 32, device: str = "cpu"
    ) -> None:
        if not model_path.is_dir():
            raise NotADirectoryError(model_path)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_id = model_path.name
        self.batch_size = batch_size
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for NLI but is unavailable")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True
        ).to(self.device).eval()
        label_map = {
            str(name).lower(): int(index)
            for index, name in self.model.config.id2label.items()
        }
        matches = [index for name, index in label_map.items() if "entail" in name]
        if len(matches) != 1:
            raise ValueError(f"cannot identify entailment label from {label_map}")
        self.entailment_label = matches[0]

    def judge(self, premise: str, hypothesis: str) -> bool:
        return self.check_pairs([(premise, hypothesis)])[0]

    def check_pairs(self, pairs: list[tuple[str, str]]) -> list[bool]:
        import torch

        results: list[bool] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            encoded = self.tokenizer(
                [premise for premise, _ in batch],
                [hypothesis for _, hypothesis in batch],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                labels = self.model(**encoded).logits.argmax(dim=-1).tolist()
            results.extend(label == self.entailment_label for label in labels)
        return results
