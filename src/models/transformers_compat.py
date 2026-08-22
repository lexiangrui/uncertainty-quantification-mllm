"""Compatibility shims for original InternVL remote model code."""

from __future__ import annotations


def patch_tied_weights_keys_compat() -> None:
    """Bridge the old and new Transformers tied-weight attribute names."""
    from transformers.modeling_utils import PreTrainedModel

    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    def getter(model):
        return getattr(model, "_tied_weights_keys", None) or {}

    def setter(model, value) -> None:
        model._tied_weights_keys = value or {}

    PreTrainedModel.all_tied_weights_keys = property(getter, setter)
