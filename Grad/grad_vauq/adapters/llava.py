"""LLaVA visual-token adapter."""

from __future__ import annotations

from contextlib import contextmanager

from .base import VisionTokenAdapter
from ..types import VisualTokenTrace


class LlavaVisualTokenAdapter(VisionTokenAdapter):
    """Hook LLaVA's multimodal projector output.

    Hugging Face LLaVA sends vision-tower features through a multimodal
    projector before inserting them into the language-model sequence. Depending
    on the Transformers version, this module lives at either
    `multi_modal_projector` or `model.multi_modal_projector`.
    """

    def _projector(self, model):
        if hasattr(model, "multi_modal_projector"):
            return model.multi_modal_projector, "multi_modal_projector"
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "multi_modal_projector"):
            return inner.multi_modal_projector, "model.multi_modal_projector"
        raise AttributeError(
            "Expected LLaVA model to expose `multi_modal_projector` or "
            "`model.multi_modal_projector`."
        )

    @contextmanager
    def capture(self, model):
        holder = {}

        projector, projector_name = self._projector(model)

        def hook(_module, _args, output):
            features = output.detach().clone().requires_grad_(True)
            holder["trace"] = VisualTokenTrace(
                features=features,
                spatial_shape=self.infer_spatial_shape(features.shape[1]),
                metadata={"adapter": "llava", "hook": projector_name},
            )
            return features

        handle = projector.register_forward_hook(hook)
        try:
            yield holder
        finally:
            handle.remove()
