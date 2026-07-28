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
            dtype=torch_dtype or torch.float16,
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
    ) -> str:
        answer = self.generate_batch(
            [image], [question], temp=temp, max_new_tokens=max_new_tokens
        )[0]
        return answer

    @torch.no_grad()
    def generate_batch(
        self,
        images: list,
        questions: list[str],
        temp: float = 0.1,
        max_new_tokens: int = 64,
    ) -> list[str]:
        if len(images) != len(questions):
            raise ValueError("images and questions must have the same length")
        if not images:
            return []
        normalized_images = [
            Image.open(image).convert("RGB") if isinstance(image, str) else image.convert("RGB")
            for image in images
        ]
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image", "image": image},
                    ],
                }
            ]
            for image, question in zip(normalized_images, questions)
        ]
        prompts = [
            self.processor.apply_chat_template(item, add_generation_prompt=True)
            for item in conversations
        ]
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inputs = self.processor(
            text=prompts,
            images=normalized_images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        self.tokenizer.padding_side = original_padding_side
        prompt_width = inputs.input_ids.shape[1]
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            pad_token_id=self.tokenizer.eos_token_id,
        )[:, prompt_width:]
        return [
            text.strip()
            for text in self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ]
