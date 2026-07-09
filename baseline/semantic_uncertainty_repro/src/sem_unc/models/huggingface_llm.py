"""HuggingFace causal LM backend for text-only semantic uncertainty.

Supports LLaMA-2, Falcon, Mistral, and similar models via the
``AutoModelForCausalLM`` interface.
"""

from __future__ import annotations

import logging
from typing import List

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from .base import Model

logger = logging.getLogger(__name__)

STOP_SEQUENCES = ["\n\n\n\n", "\n\n\n", "\n\n", "\n", "Question:", "Context:"]


class _StoppingCriteriaSub(StoppingCriteria):
    """Stop generation when a stop-sequence appears in the decoded text."""

    def __init__(self, stops: list[str], tokenizer, initial_length: int):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> bool:
        del scores
        generation = self.tokenizer.decode(
            input_ids[0][self.initial_length :], skip_special_tokens=False
        )
        return any(stop in generation for stop in self.stops)


class HuggingFaceLLM(Model):
    """Text-only causal LM for semantic uncertainty.

    Parameters
    ----------
    model_path:
        HuggingFace repo id (e.g. ``"meta-llama/Llama-2-7b-chat-hf"``) or
        local directory.
    max_new_tokens:
        Maximum number of tokens to generate per answer.
    load_in_8bit:
        If ``True``, load with 8-bit quantization to reduce VRAM.
    """

    def __init__(
        self,
        model_path: str = "meta-llama/Llama-2-7b-chat-hf",
        max_new_tokens: int = 50,
        load_in_8bit: bool = False,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        model_name = model_path

        # --- tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            device_map="auto",
            token_type_ids=None,
            clean_up_tokenization_spaces=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # --- model ---
        kwargs: dict = {}
        if load_in_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        self.model.eval()

        self._stop_sequences = STOP_SEQUENCES + [self.tokenizer.eos_token]
        self._token_limit = getattr(self.model.config, "max_position_embeddings", 4096)

        logger.info(
            "Loaded %s on %s (max_new_tokens=%d, 8bit=%s)",
            model_name,
            self.device,
            max_new_tokens,
            load_in_8bit,
        )

    # ------------------------------------------------------------------
    # Model ABC
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        question: str,
        image: Image.Image | None = None,
        temperature: float = 0.1,
        max_new_tokens: int | None = None,
    ) -> tuple[str, list[float], torch.Tensor | None]:
        """Sample one answer from a text-only prompt.

        The *image* parameter is ignored; it exists for interface
        compatibility with multimodal models.
        """
        max_tokens = max_new_tokens or self.max_new_tokens

        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        n_input = inputs.input_ids.shape[1]
        stopping_criteria = StoppingCriteriaList(
            [
                _StoppingCriteriaSub(
                    stops=self._stop_sequences,
                    tokenizer=self.tokenizer,
                    initial_length=n_input,
                )
            ]
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            temperature=temperature,
            do_sample=True,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # --- decode answer ---
        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )

        # Strip the input prompt from the generated text.
        input_text = self.tokenizer.decode(
            inputs.input_ids[0], skip_special_tokens=True
        )
        if full_answer.startswith(input_text):
            raw_answer = full_answer[len(input_text) :]
        else:
            raw_answer = full_answer

        stop_at = len(raw_answer)
        for stop in self._stop_sequences:
            pos = raw_answer.find(stop)
            if pos != -1:
                stop_at = min(stop_at, pos)
        answer = raw_answer[:stop_at].strip()

        # --- token log-likelihoods ---
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        token_stop_index = self.tokenizer(
            input_text + raw_answer[:stop_at], return_tensors="pt"
        )["input_ids"].shape[1]
        n_generated = max(token_stop_index - n_input, 1)
        log_likelihoods = transition_scores[0].tolist()[:n_generated]

        if n_generated == 0:
            logger.warning(
                "Zero tokens generated for question: %s", question[:80]
            )
            return answer, [], None

        # --- last-token hidden state ---
        hidden_states = outputs.hidden_states
        if hidden_states and len(hidden_states) > 0:
            if len(hidden_states) >= n_generated:
                idx = n_generated - 1
            else:
                idx = -1
            last_step = hidden_states[idx]
            last_hidden = last_step[-1][:, -1, :].cpu()
        else:
            last_hidden = None

        return answer, log_likelihoods, last_hidden
