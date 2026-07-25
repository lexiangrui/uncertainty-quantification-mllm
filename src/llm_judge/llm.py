"""One project-wide Qwen judge for free-form VQA answers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def build_judge_prompt(question: str, references: list[str], prediction: str) -> str:
    payload = {
        "task": (
            "Judge whether the model answer is semantically correct for the question according to "
            "the reference answers. Ignore harmless wording differences. Mark WRONG if it "
            "contradicts a reference, misses a required fact, or adds a false claim. Return exactly "
            "one JSON object and no other text: {\"verdict\":\"CORRECT\"} or "
            "{\"verdict\":\"WRONG\"}."
        ),
        "question": question,
        "reference_answers": [str(item) for item in references],
        "model_answer": prediction,
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_judge_verdict(text: str) -> bool:
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as error:
        raise ValueError(f"judge returned invalid JSON: {text[:200]!r}") from error
    if not isinstance(parsed, dict) or set(parsed) != {"verdict"}:
        raise ValueError(f"judge returned invalid object: {text[:200]!r}")
    verdict = parsed["verdict"]
    if verdict not in {"CORRECT", "WRONG"}:
        raise ValueError(f"judge returned invalid verdict: {text[:200]!r}")
    return verdict == "CORRECT"


class QwenLLMJudge:
    """Greedy Qwen judge; may reuse an already-loaded tokenizer and model."""

    name = "qwen_llm"

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        tokenizer=None,
        model=None,
        device: str = "cuda:0",
        local_files_only: bool = True,
    ):
        if (tokenizer is None) != (model is None):
            raise ValueError("tokenizer and model must be supplied together")
        if model is None:
            if model_path is None:
                raise ValueError("model_path is required when model is not supplied")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map={"": device},
                local_files_only=local_files_only,
            ).eval()
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.device = next(self.model.parameters()).device
        self.last_result: dict[str, Any] | None = None

    @torch.inference_mode()
    def judge(self, question: str, references: list[str], prediction: str) -> bool:
        prompt = build_judge_prompt(question, references, prediction)
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        encoded = self.tokenizer(rendered, return_tensors="pt").to(self.device)
        output = self.model.generate(**encoded, do_sample=False, max_new_tokens=32)
        response = self.tokenizer.decode(
            output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        correct = parse_judge_verdict(response)
        self.last_result = {"correct": correct, "raw_response": response}
        return correct
