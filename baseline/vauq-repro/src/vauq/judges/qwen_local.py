"""Local Qwen text judge for free-form VQA answers."""

from __future__ import annotations

import os
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import Judge


class QwenLocalJudge(Judge):
    def __init__(self):
        self.model_path = os.environ["QWEN_JUDGE_MODEL"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.last_result: dict[str, Any] | None = None

    def judge(self, prediction: str, gold: Any, sample: dict[str, Any] | None = None) -> bool:
        prompt = _build_prompt(prediction, gold, sample or {})
        messages = [
            {"role": "system", "content": "You are a strict but fair VQA judge. Reply only CORRECT or WRONG."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        new_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        correct = _parse_verdict(response)
        self.last_result = {
            "correct": correct,
            "raw_response": response[:1000],
            "model_path": self.model_path,
        }
        return correct


def _build_prompt(prediction: str, gold: Any, sample: dict[str, Any]) -> str:
    return (
        f"Question: {sample.get('question') or ''}\n"
        f"Ground truth: {gold}\n"
        f"Model answer: {prediction}\n"
        "Does the model answer match the ground truth? Reply with CORRECT or WRONG only."
    )


def _parse_verdict(text: str) -> bool:
    match = re.search(r"\b(INCORRECT|WRONG|FALSE|CORRECT|TRUE)\b", text.upper())
    if not match:
        raise ValueError(f"Local judge returned unparsable verdict: {text[:200]!r}")
    return match.group(1) in {"CORRECT", "TRUE"}
