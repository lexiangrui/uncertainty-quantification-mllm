"""Model registry and factory."""

from __future__ import annotations

from .base import Model
from .huggingface_llm import HuggingFaceLLM
from .llava import LlavaModel

MODEL_MAP: dict[str, type[Model]] = {
    "llava": LlavaModel,
    "huggingface_llm": HuggingFaceLLM,
}


def build_model(name: str, **kwargs) -> Model:
    """Instantiate a model by name.

    Parameters
    ----------
    name:
        Registry key (currently only ``"llava"``).
    **kwargs:
        Forwarded to the model constructor.
    """
    cls = MODEL_MAP.get(name)
    if cls is None:
        raise KeyError(
            f"Unknown model '{name}'. Available: {list(MODEL_MAP)}"
        )
    return cls(**kwargs)
