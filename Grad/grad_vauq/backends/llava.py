"""LLaVA backend for Grad-VAUQ."""

from __future__ import annotations

import os

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..adapters import build_adapter


def _resolve_model_name(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    if "/" in model_path:
        return model_path
    return f"llava-hf/{model_path}"


class GradLlavaBackend:
    """LLaVA-1.5 backend using gradient visual-token attribution."""

    def __init__(
        self,
        model_path: str = "llava-hf/llava-1.5-7b-hf",
        device: str | None = None,
        torch_dtype=None,
        attn_implementation: str = "flash_attention_2",
        adapter: str = "llava",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.attn_implementation = attn_implementation
        model_name = _resolve_model_name(model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype or torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        self.adapter = build_adapter(adapter)

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

    def prepare_full_inputs(self, image, question, generated_ids):
        inputs, _ = self._prepare_inputs(image, question)
        full_ids = torch.cat([inputs.input_ids, generated_ids.to(self.device)], dim=1)
        attention_mask = torch.cat(
            [
                inputs.attention_mask,
                torch.ones_like(generated_ids.to(self.device)),
            ],
            dim=1,
        )
        return inputs, full_ids, attention_mask, inputs.input_ids.shape[1]

    def generate_with_ids(self, image, question, temp: float = 0.0, max_new_tokens: int = 128):
        inputs, _ = self._prepare_inputs(image, question)
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temp > 0,
        }
        if temp > 0:
            generate_kwargs["temperature"] = temp
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                **generate_kwargs,
            )
        generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
        answer = self.processor.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return answer, generated_ids

    def forward_logits(self, image, question, generated_ids):
        inputs, full_ids, attention_mask, prompt_len = self.prepare_full_inputs(
            image, question, generated_ids
        )
        with torch.no_grad():
            outputs = self.model(
                input_ids=full_ids,
                pixel_values=inputs.pixel_values,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return outputs.logits, prompt_len

    def forward_logits_with_trace(self, image, question, generated_ids):
        inputs, full_ids, attention_mask, prompt_len = self.prepare_full_inputs(
            image, question, generated_ids
        )
        with self.adapter.capture(self.model) as holder:
            outputs = self.model(
                input_ids=full_ids,
                pixel_values=inputs.pixel_values,
                attention_mask=attention_mask,
                return_dict=True,
            )
        if "trace" not in holder:
            raise RuntimeError("Visual adapter did not capture visual token features.")
        return outputs.logits, prompt_len, holder["trace"]

    def forward_logits_with_ablation(
        self,
        image,
        question,
        generated_ids,
        selected_indices: torch.Tensor,
        baseline: str = "zero",
    ):
        inputs, full_ids, attention_mask, prompt_len = self.prepare_full_inputs(
            image, question, generated_ids
        )
        with torch.no_grad():
            with self.adapter.ablate(self.model, selected_indices, baseline=baseline):
                outputs = self.model(
                    input_ids=full_ids,
                    pixel_values=inputs.pixel_values,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        return outputs.logits, prompt_len

    def forward_logits_with_visual_features(
        self,
        image,
        question,
        generated_ids,
        features: torch.Tensor,
    ):
        inputs, full_ids, attention_mask, prompt_len = self.prepare_full_inputs(
            image, question, generated_ids
        )
        if features.shape[0] != full_ids.shape[0]:
            batch_size = features.shape[0]
            full_ids = full_ids.expand(batch_size, -1)
            attention_mask = attention_mask.expand(batch_size, -1)
            inputs.pixel_values = inputs.pixel_values.expand(batch_size, *inputs.pixel_values.shape[1:])
        with self.adapter.override(self.model, features):
            outputs = self.model(
                input_ids=full_ids,
                pixel_values=inputs.pixel_values,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return outputs.logits, prompt_len
