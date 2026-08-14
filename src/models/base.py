from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from PIL import Image

from src.generation.prompt import GenerationPrompt


@dataclass(frozen=True)
class GeneratedResponse:
    text: str
    token_ids: tuple[int, ...]
    token_log_probs: tuple[float, ...]
    sampling_token_log_probs: tuple[float, ...]
    final_hidden: tuple[float, ...] = ()
    hidden_steps: tuple[tuple[float, ...], ...] = ()
    finish_reason: str | None = None
    rng_seed: int | None = None


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    sample_id: str
    role: Literal["greedy", "sample"]
    draw_index: int | None
    seed: int
    image: Image.Image | None
    prompt: GenerationPrompt


class GenerationBackend(ABC):
    model_id: str

    @property
    def runtime_config(self) -> dict:
        return {}

    @abstractmethod
    def decode_generated_tokens(self, token_ids: tuple[int, ...]) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_requests(
        self, requests: list[GenerationRequest], *, max_new_tokens: int
    ) -> dict[str, GeneratedResponse]:
        raise NotImplementedError
