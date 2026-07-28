"""LLaVA-1.5 backend."""

from __future__ import annotations

import os

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ..masking import choose_masked_offsets
from .base import Backend


def _resolve_model_name(model_path: str) -> str:
    """Accept either a local directory, a full repo id, or a short LLaVA suffix."""
    if os.path.isdir(model_path):
        return model_path
    if "/" in model_path:
        return model_path
    return f"llava-hf/{model_path}"


class LlavaBackend(Backend):
    """LLaVA-1.5 with VAUQ core-region attention masking."""

    def __init__(
        self,
        model_path: str = "llava-hf/llava-1.5-7b-hf",
        device: str | None = None,
        torch_dtype=None,
        attn_implementation: str = "eager",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.attn_implementation = attn_implementation
        model_name = _resolve_model_name(model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch_dtype or torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
        ).to(self.device)
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

    def generate_with_ids(self, image, question, max_new_tokens: int = 128):
        inputs, _ = self._prepare_inputs(image, question)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        answer = self.processor.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return answer, generated_ids

    def get_logits(self, image, question, generated_ids):
        inputs, _ = self._prepare_inputs(image, question)
        full_ids = torch.cat([inputs.input_ids, generated_ids], dim=1)

        with torch.no_grad():
            outputs = self.model(
                input_ids=full_ids,
                pixel_values=inputs.pixel_values,
                return_dict=True,
            )

        prompt_len = inputs.input_ids.shape[1]
        return outputs.logits[0, prompt_len - 1: -1]

    def get_logits_masked(
        self,
        image,
        question,
        generated_ids,
        topk_ratio: float = 0.6,
        layer_range: tuple[int, int] = (10, 25),
        mask_strategy: str = "core",
        mask_seed: int | None = None,
    ):
        """Forward pass with core, equal-count random, or all-token masking."""
        inputs, _ = self._prepare_inputs(image, question)
        full_ids = torch.cat([inputs.input_ids, generated_ids], dim=1)
        prompt_len = inputs.input_ids.shape[1]

        vision_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        positions = (full_ids[0] == vision_token_id).nonzero(as_tuple=True)[0]
        if positions.numel() != 576:
            raise AssertionError(
                f"VAUQ requires 576 expanded LLaVA visual tokens, got {positions.numel()}"
            )
        first_pos, last_pos = positions[0].item(), positions[-1].item() + 1
        if last_pos - first_pos != positions.numel():
            raise AssertionError("expanded LLaVA visual tokens must form one contiguous span")

        selected_vis_attentions = None
        if mask_strategy == "core":
            with torch.no_grad():
                outputs = self.model(
                    input_ids=full_ids,
                    pixel_values=inputs.pixel_values,
                    output_attentions=True,
                    return_dict=True,
                )
            attentions = torch.stack(outputs.attentions, dim=0).squeeze(1)
            selected_vis_attentions = (
                attentions[layer_range[0]: layer_range[1]][
                    :, :, prompt_len:, first_pos:last_pos
                ]
                .mean(0)
                .mean(0)
                .mean(0)
            )
            if selected_vis_attentions.numel() != positions.numel():
                raise AssertionError(
                    f"visual attention length {selected_vis_attentions.numel()} does not "
                    f"match image-token count {positions.numel()}"
                )

        masked_offsets = choose_masked_offsets(
            positions.numel(),
            topk_ratio,
            mask_strategy,
            attention_scores=selected_vis_attentions,
            seed=mask_seed,
        ).to(full_ids.device)

        attention_mask = torch.cat(
            [inputs.attention_mask, torch.ones_like(generated_ids)], dim=1
        )
        absolute_masked_indices = first_pos + masked_offsets
        attention_mask[0, absolute_masked_indices] = 0
        with torch.no_grad():
            outputs_masked = self.model(
                input_ids=full_ids,
                pixel_values=inputs.pixel_values,
                attention_mask=attention_mask,
                return_dict=True,
            )

        return outputs_masked.logits[0, prompt_len - 1: -1]
