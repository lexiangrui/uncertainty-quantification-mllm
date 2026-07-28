import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from choice_consistency import INVALID_CHOICE, choice_classes, consistency_metrics
from evaluate_choice_consistency import evaluate_record, summarize
from model import LlavaMalpRunner
from perturb import PerturbSpec
from run_choice_consistency import process_one


def test_choice_classes_normalizes_chatty_answers_and_keeps_invalid():
    assert choice_classes(["A", "The answer is B.", "Option C", "unknown"], 3) == [
        "A",
        "B",
        "C",
        INVALID_CHOICE,
    ]


def test_consistency_metrics_all_same_is_zero_uncertainty():
    metrics = consistency_metrics(["A", "A", "A", "A"], "A")
    assert metrics["answer_entropy"] == 0.0
    assert metrics["variation_ratio"] == 0.0
    assert metrics["base_flip_rate"] == 0.0
    assert metrics["pairwise_disagreement"] == 0.0


def test_consistency_metrics_two_balanced_classes():
    metrics = consistency_metrics(["A", "A", "B", "B"], "A")
    assert math.isclose(metrics["answer_entropy"], 0.5)
    assert metrics["variation_ratio"] == 0.5
    assert metrics["base_flip_rate"] == 0.5
    assert math.isclose(metrics["pairwise_disagreement"], 4 / 6)


def test_consistency_metrics_requires_multiple_generations():
    with pytest.raises(ValueError, match="at least two"):
        consistency_metrics(["A"], "A")


def _record(prediction="A"):
    return {
        "id": "choice-1",
        "dataset": "cvbench2d",
        "prediction": prediction,
        "choices": ["red", "blue"],
        "answer_index": 0,
        "generations": [
            {"stage": "fusion", "mode": "directional", "seed": 42, "sigma": 0.1, "text": "A"},
            {"stage": "fusion", "mode": "directional", "seed": 43, "sigma": 0.1, "text": "B"},
            {"stage": "fusion", "mode": "directional", "seed": 44, "sigma": 0.1, "text": "B"},
        ],
    }


def test_evaluate_record_computes_choice_consistency():
    result = evaluate_record(_record())[0]
    assert result["base_choice"] == "A"
    assert result["gold_choice"] == "A"
    assert result["correct"] is True
    assert result["class_counts"] == {"A": 1, "B": 2}
    assert math.isclose(result["variation_ratio"], 1 / 3)
    assert math.isclose(result["base_flip_rate"], 2 / 3)


def test_summarize_reports_error_auroc():
    correct = evaluate_record(_record("A"))[0]
    wrong_record = _record("B")
    wrong_record["id"] = "choice-2"
    wrong_record["generations"] = [
        {**item, "text": text}
        for item, text in zip(wrong_record["generations"], ["A", "A", "A"], strict=True)
    ]
    wrong = evaluate_record(wrong_record)[0]
    result = summarize([correct, wrong])[0]
    assert result["num_samples"] == 2
    assert "auroc_answer_entropy_error" in result


def test_process_one_rejects_open_ended_samples():
    with pytest.raises(ValueError, match="not multiple choice"):
        process_one(
            {"id": "open", "choices": None, "answer_index": None},
            object(),
            modes=["directional"],
            stages=["fusion"],
            seeds=(42, 43),
            sigma=0.1,
            gamma=1.0,
            experiment_config={},
        )


def test_generation_path_rejects_adversarial_mode():
    runner = object.__new__(LlavaMalpRunner)
    spec = PerturbSpec("joint", "fusion", "adversarial", 0.1, 1.0, 42)
    inputs = {
        "input_ids": torch.tensor([[1, 2]]),
        "question_token_mask": torch.tensor([[False, True]]),
    }
    with pytest.raises(ValueError, match="does not support adversarial"):
        runner._register_generation_perturb_hooks(inputs, spec)


def test_process_one_records_each_perturbed_generation():
    class FakeRunner:
        def prepare_inputs(self, _image, _question):
            return {"input_ids": torch.zeros(1, 2, dtype=torch.long)}

        def greedy_generate(self, _inputs):
            ids = torch.tensor([[1]])
            return {"text": "A", "answer_ids": ids, "answer_mask": torch.ones_like(ids).bool()}

        def generate_with_perturbation(self, _inputs, spec):
            return {"text": "A" if spec.seed == 42 else "B", "answer_ids": torch.tensor([[spec.seed]])}

    sample = {
        "id": "choice",
        "dataset": "cvbench2d",
        "image": object(),
        "question": "question",
        "references": ["A"],
        "choices": ["yes", "no"],
        "answer_index": 0,
        "metadata": {},
    }
    result = process_one(
        sample,
        FakeRunner(),
        modes=["directional"],
        stages=["fusion"],
        seeds=(42, 43),
        sigma=0.1,
        gamma=1.0,
        experiment_config={},
    )
    assert [item["text"] for item in result["generations"]] == ["A", "B"]
    assert [item["seed"] for item in result["generations"]] == [42, 43]
