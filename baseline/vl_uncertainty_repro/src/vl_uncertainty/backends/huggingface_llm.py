"""Text-only HuggingFace LLM backend for semantic uncertainty verification."""

from __future__ import annotations

import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from .base import Backend

logger = logging.getLogger(__name__)

STOP_SEQUENCES = ["\n\n\n\n", "\n\n\n", "\n\n", "\n", "Question:", "Context:"]


class _StopCriteria(StoppingCriteria):
    def __init__(self, stops: list[str], tokenizer, initial_length: int):
        super().__init__()
        self.stops = stops
        self.initial_length = initial_length
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        del scores
        generation = self.tokenizer.decode(
            input_ids[0][self.initial_length :], skip_special_tokens=False
        )
        return any(stop in generation for stop in self.stops)


class HuggingFaceLLMBackend(Backend):
    """Text-only causal LM backend.

    Ignores the *image* argument (exists for interface compatibility).
    """

    def __init__(
        self,
        model_path: str = "mistralai/Mistral-7B-Instruct-v0.1",
        load_in_8bit: bool = False,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, clean_up_tokenization_spaces=False
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        kwargs = {}
        if load_in_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        self.model.eval()

        self._stop_seqs = STOP_SEQUENCES + [self.tokenizer.eos_token]
        logger.info("Loaded HuggingFace LLM: %s", model_path)

    @torch.no_grad()
    def generate(
        self,
        image=None,  # ignored for text-only
        question: str = "",
        temp: float = 0.1,
        max_new_tokens: int = 64,
    ) -> tuple[str, list[float], torch.Tensor | None]:
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        n_input = inputs.input_ids.shape[1]
        stop_criteria = StoppingCriteriaList(
            [_StopCriteria(self._stop_seqs, self.tokenizer, n_input)]
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            temperature=temp,
            do_sample=True,
            stopping_criteria=stop_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        full_answer = self.tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )
        input_text = self.tokenizer.decode(
            inputs.input_ids[0], skip_special_tokens=True
        )
        answer = (
            full_answer[len(input_text) :].strip()
            if full_answer.startswith(input_text)
            else full_answer.strip()
        )

        # Token log-likelihoods.
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        n_gen = transition_scores.shape[1]
        log_likelihoods = transition_scores[0].tolist() if n_gen > 0 else []

        # Last-token embedding.
        hidden_states = outputs.hidden_states
        if hidden_states and len(hidden_states) > 0 and n_gen > 0:
            idx = min(n_gen - 1, len(hidden_states) - 1)
            last_hidden = hidden_states[idx][-1][:, -1, :].cpu()
        else:
            last_hidden = None

        return answer, log_likelihoods, last_hidden
