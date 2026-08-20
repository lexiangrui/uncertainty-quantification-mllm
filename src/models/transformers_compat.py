"""Small compatibility shims for remote model implementations.

The OpenGVLab InternVL3.5 remote code targets the Transformers API that used
``_tied_weights_keys``.  Newer Transformers releases expose the public
``all_tied_weights_keys`` name during checkpoint finalization.  Add a
read/write bridge only when the installed version lacks that attribute, so
other model families keep their native behavior.
"""

from __future__ import annotations


def patch_tied_weights_keys_compat() -> None:
    """Bridge the old and new tied-weight attribute names, if necessary."""

    from transformers.modeling_utils import PreTrainedModel

    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    def getter(model):
        return getattr(model, "_tied_weights_keys", None) or {}

    def setter(model, value) -> None:
        model._tied_weights_keys = value or {}

    PreTrainedModel.all_tied_weights_keys = property(getter, setter)
