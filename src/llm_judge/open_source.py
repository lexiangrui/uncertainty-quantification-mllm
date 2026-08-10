"""Open-source Qwen judge selector used by the project.

The text and vision backends are kept separate because they have different
input/output contracts, while callers select them through one model-name API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .qwen_vl import QwenVLJudge
from .qwen_text import QwenTextJudge


QWEN_TEXT_MODELS = frozenset(
    {
        "qwen2.5-3b-instruct",
        "qwen3-4b-instruct-2507",
    }
)
QWEN_VL_MODELS = frozenset({"qwen3.6-vl"})


def _canonical_model_name(model_name: str) -> str:
    return Path(model_name.rstrip("/")).name.lower()


def _backend_kind(model_name: str) -> str:
    canonical = _canonical_model_name(model_name)
    if canonical in QWEN_TEXT_MODELS:
        return "text"
    if canonical in QWEN_VL_MODELS:
        return "vl"
    if "qwen" not in canonical:
        raise ValueError(
            f"unsupported open-source judge model {model_name!r}; "
            "expected a registered Qwen model"
        )
    if "vl" in canonical or "multimodal" in canonical:
        return "vl"
    return "text"


class OpenSourceJudge:
    """Facade for the two supported local Qwen judges.

    ``model_name`` selects the implementation. Registered names are
    ``Qwen2.5-3B-Instruct`` and ``Qwen3-4B-Instruct-2507`` for the text judge,
    and ``Qwen3.6-VL`` for the multimodal MMHal judge. ``model_path`` can be
    supplied separately when the display name and local checkpoint path differ.
    """

    def __init__(
        self,
        model_name: str,
        *,
        model_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")
        self.model_name = model_name
        checkpoint = model_path or model_name
        kind = _backend_kind(model_name)
        if kind == "vl":
            self.backend = QwenVLJudge(checkpoint, **kwargs)
            self.kind = "qwen3.6-vl"
        else:
            self.backend = QwenTextJudge(checkpoint, **kwargs)
            self.kind = _canonical_model_name(model_name)

    @property
    def name(self) -> str:
        return f"open_source:{self.model_name}"

    @property
    def last_result(self) -> Any:
        return self.backend.last_result

    def judge(self, *args: Any, **kwargs: Any) -> Any:
        return self.backend.judge(*args, **kwargs)
