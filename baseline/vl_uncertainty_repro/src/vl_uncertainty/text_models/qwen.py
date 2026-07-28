"""Qwen text backend used for rephrasing and free-form entailment."""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor

from .base import TextModel


class _PerRowTemperature(LogitsProcessor):
    def __init__(self, temperatures: list[float]):
        if any(temp <= 0 for temp in temperatures):
            raise ValueError("all temperatures must be positive")
        self.temperatures = temperatures

    def __call__(self, input_ids, scores):
        temps = torch.tensor(self.temperatures, device=scores.device, dtype=scores.dtype)
        return scores / temps[:, None]


def _resolve_model_name(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    if "/" in model_path:
        return model_path
    return f"Qwen/{model_path}"


class QwenTextModel(TextModel):
    def __init__(self, model_path: str = "Qwen2.5-3B-Instruct", device: str = "cuda:0"):
        model_name = _resolve_model_name(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map={"": device},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompt: str, temp: float = 0.1, max_new_tokens: int = 256) -> str:
        return self.generate_batch([prompt], [temp], max_new_tokens=max_new_tokens)[0]

    def generate_batch(
        self, prompts: list[str], temps: list[float], max_new_tokens: int = 256
    ) -> list[str]:
        if len(prompts) != len(temps):
            raise ValueError("prompts and temps must have the same length")
        if not prompts:
            return []
        messages = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            for prompt in prompts
        ]
        texts = [
            self.tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
            for item in messages
        ]
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
        self.tokenizer.padding_side = original_padding_side
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.8,
                repetition_penalty=1.05,
                logits_processor=[_PerRowTemperature(temps)],
            )
        prompt_width = model_inputs.input_ids.shape[1]
        answer_ids = generated_ids[:, prompt_width:]
        return [
            text.strip()
            for text in self.tokenizer.batch_decode(answer_ids, skip_special_tokens=True)
        ]
