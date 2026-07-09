"""LLaVA visual-token adapter."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .base import VisionTokenAdapter
from ..types import VisualTokenTrace


class LlavaVisualTokenAdapter(VisionTokenAdapter):
    """Hook LLaVA's multimodal projector output.

    Hugging Face LLaVA sends vision-tower features through
    `model.multi_modal_projector` before inserting them into the language-model
    sequence. That projector output is a stable visual-token boundary for both
    attribution and ablation.
    """

    def _projector(self, model):
        if not hasattr(model, "multi_modal_projector"):
            raise AttributeError("Expected LLaVA model to expose `multi_modal_projector`.")
        return model.multi_modal_projector

    @contextmanager
    def capture(self, model):
        holder = {}

        def hook(_module, _args, output):
            features = output.detach().clone().requires_grad_(True)
            holder["trace"] = VisualTokenTrace(
                features=features,
                spatial_shape=self.infer_spatial_shape(features.shape[1]),
                metadata={"adapter": "llava", "hook": "multi_modal_projector"},
            )
            return features

        handle = self._projector(model).register_forward_hook(hook)
        try:
            yield holder
        finally:
            handle.remove()

    @contextmanager
    def ablate(self, model, indices: torch.Tensor, baseline: str = "zero"):
        index_cpu = indices.detach().long().cpu()

        def hook(_module, _args, output):
            modified = output.clone()
            index = index_cpu.to(modified.device)
            if baseline == "zero":
                replacement = torch.zeros_like(modified[:, index, :])
            elif baseline == "mean":
                replacement = modified.mean(dim=1, keepdim=True).expand(-1, index.numel(), -1)
            else:
                raise ValueError("Unsupported ablation baseline. Use 'zero' or 'mean'.")
            modified[:, index, :] = replacement
            return modified

        handle = self._projector(model).register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()
