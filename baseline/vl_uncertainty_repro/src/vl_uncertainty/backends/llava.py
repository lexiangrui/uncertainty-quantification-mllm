"""LLaVA backend — returns token log-likelihoods for semantic entropy."""

from __future__ import annotations

import logging
import os

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .base import Backend

logger = logging.getLogger(__name__)


def _resolve_model_name(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    if "/" in model_path:
        return model_path
    return f"llava-hf/{model_path}"


class LlavaBackend(Backend):
    """LLaVA-1.5 backend with token log-likelihood extraction."""

    def __init__(
        self,
        model_path: str = "llava-hf/llava-1.5-7b-hf",
        device: str | None = None,
        torch_dtype=None,
        attn_implementation: str = "eager",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_name = _resolve_model_name(model_path)

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype or torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        ).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer

    def _prepare_inputs(self, image, question):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image", "image": image},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.device)
        return inputs, image

    # ------------------------------------------------------------------
    # Backend ABC
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        image,
        question: str,
        temp: float = 0.1,
        max_new_tokens: int = 64,
    ) -> tuple[str, list[float], torch.Tensor | None]:
        inputs, _ = self._prepare_inputs(image, question)
        n_input = inputs.input_ids.shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = outputs.sequences[:, n_input:]

        answer = self.processor.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        # Token log-likelihoods.
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        n_gen = transition_scores.shape[1]
        log_likelihoods = transition_scores[0].tolist() if n_gen > 0 else []

        # Last-token embedding.
        hidden_states = outputs.hidden_states
        if hidden_states and len(hidden_states) > 0 and n_gen > 0:
            last_hidden = hidden_states[-1][-1][:, -1, :].cpu()
        else:
            last_hidden = None

        return answer, log_likelihoods, last_hidden
