"""Prompt helpers and evaluation utilities."""

from __future__ import annotations

import re
import string


def parse_original_question(question: str) -> str:
    return question.split("\n")[0]


def merge_question(question_new: str, question: str) -> str:
    suffix = "\n".join(question.split("\n")[1:])
    return question_new if not suffix else f"{question_new}\n{suffix}"


def normalize_answer(s: str) -> str:
    """Lower-case, remove articles, punctuation, and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def squad_f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)
