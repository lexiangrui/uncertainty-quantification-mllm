from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

from src.generation.prompt import GenerationPrompt


@dataclass(frozen=True)
class GeneratedResponse:
    text: str
    token_ids: tuple[int, ...]
    token_log_probs: tuple[float, ...]
    sampling_token_log_probs: tuple[float, ...]
    final_hidden: tuple[float, ...]


class GenerationBackend(ABC):
    model_id: str

    @property
    def runtime_config(self) -> dict:
        return {}

    @abstractmethod
    def decode_generated_tokens(self, token_ids: tuple[int, ...]) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        image: Image.Image | None,
        prompt: GenerationPrompt,
        *,
        do_sample: bool,
        temperature: float | None,
        max_new_tokens: int,
        num_return_sequences: int,
    ) -> list[GeneratedResponse]:
        raise NotImplementedError
