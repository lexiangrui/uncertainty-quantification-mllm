"""LLaVA model backend for semantic uncertainty."""

from __future__ import annotations

import logging
import os

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .base import Model


def _resolve_model_name(model_path: str) -> str:
    """Accept a local directory, a full repo id, or a short LLaVA suffix."""
    if os.path.isdir(model_path):
        return model_path
    if "/" in model_path:
        return model_path
    return f"llava-hf/{model_path}"


class LlavaModel(Model):
    """LLaVA-1.5 generation backend for semantic uncertainty.

    Loads the model in fp16 with eager attention (required for
    ``output_hidden_states``) and exposes a single ``generate`` method
    that returns text, token log-likelihoods, and the last-token hidden
    state from the final layer.
    """

    def __init__(
        self,
        model_path: str = "llava-hf/llava-1.5-7b-hf",
        device: str | None = None,
        torch_dtype=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_name = _resolve_model_name(model_path)

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype or torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer

    def _prepare_inputs(
        self, image: Image.Image, question: str
    ) -> tuple[dict, Image.Image]:
        """Build the chat-template conversation and tokenize."""
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
    # Model ABC
    # ------------------------------------------------------------------
    def generate(
        self,
        question: str,
        image: Image.Image | None = None,
        temperature: float = 0.1,
        max_new_tokens: int = 64,
    ) -> tuple[str, list[float], torch.Tensor | None]:
        """Sample one answer, returning (text, log_liks, last_hidden)."""
        if image is None:
            raise ValueError("LlavaModel requires an image.")
        inputs, _ = self._prepare_inputs(image, question)
        n_input_tokens = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # --- generated token ids (excluding prompt) ---
        generated_ids = outputs.sequences[:, n_input_tokens:]

        # --- decode answer ---
        answer = self.processor.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        # --- token log-likelihoods ---
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        n_generated = transition_scores.shape[1]
        log_likelihoods = transition_scores[0].tolist()

        if n_generated == 0:
            logging.warning(
                "Zero tokens generated for question: %s", question[:80]
            )
            return answer, [], None

        # --- last-token hidden state (final layer) ---
        # outputs.hidden_states is a tuple of length n_generated.
        # Each element: tuple of (n_layers) tensors, shape
        # (batch, seq_len, hidden_size).
        # For the generated tokens, seq_len = 1 (except the first which
        # includes the prompt).
        hidden_states = outputs.hidden_states
        if len(hidden_states) > 0:
            # Take the last generated token's hidden states.
            last_step = hidden_states[-1]
            # Last layer, last token.
            last_hidden = last_step[-1][:, -1, :].cpu()
        else:
            last_hidden = None

        return answer, log_likelihoods, last_hidden
