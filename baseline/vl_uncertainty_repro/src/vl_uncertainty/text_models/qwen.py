"""Qwen text backend used for rephrasing and free-form entailment."""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import TextModel


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
            torch_dtype="auto",
            device_map={"": device},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompt: str, temp: float = 0.1, max_new_tokens: int = 256) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=0.8,
                repetition_penalty=1.05,
            )
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
