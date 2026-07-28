import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_perturb_methods import evaluate_record, roc_auc_score, summarize
from model import LlavaMalpRunner
from run_perturb import process_one


# ---- roc_auc_score ----


def test_roc_auc_perfect_separation():
    # 正样本分数更高 → AUC=1
    assert roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_roc_auc_reverse():
    # 正样本分数更低 → AUC=0
    assert roc_auc_score([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0


def test_roc_auc_all_ties_is_half():
    assert roc_auc_score([0, 1], [0.5, 0.5]) == 0.5


def test_roc_auc_single_class_returns_none():
    assert roc_auc_score([0, 0, 0], [0.1, 0.2, 0.3]) is None
    assert roc_auc_score([1, 1], [0.1, 0.2]) is None


def _brute_force_auc(labels: list[int], scores: list[float]) -> float | None:
    pos = [s for s, lab in zip(scores, labels, strict=True) if lab == 1]
    neg = [s for s, lab in zip(scores, labels, strict=True) if lab == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for positive in pos:
        for negative in neg:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def test_roc_auc_matches_brute_force():
    cases = [
        ([0, 1, 0, 1, 0, 1], [0.5, 0.5, 0.3, 0.7, 0.1, 0.9]),
        ([1, 0, 1, 0], [0.2, 0.2, 0.8, 0.9]),
        ([0, 1, 1, 0], [0.4, 0.4, 0.6, 0.6]),
        ([1, 0, 1, 0, 1], [0.9, 0.1, 0.55, 0.55, 0.55]),
    ]
    for labels, scores in cases:
        expected = _brute_force_auc(labels, scores)
        assert math.isclose(roc_auc_score(labels, scores), expected, abs_tol=1e-9), (labels, scores)


# ---- 选择题判定（经项目级 judge，不再本地解析）----


def test_evaluate_record_chatty_prediction_uses_judge():
    # 回归：旧的自写解析会把 "The answer is B" 误判成 A（命中 "answer" 里的 A）；
    # 改用 judge.RegexChoiceJudge 后应正确识别为 B → 正确。
    record = {
        "id": "chatty",
        "prediction": "The answer is B.",
        "choices": ["A", "B", "C", "D"],
        "answer_index": 1,
        "nll0": 0.5,
        "perturbations": [
            _perturbation("fusion", "norm_isotropic", 0.6, 0.01),
        ],
    }
    out = evaluate_record(record)
    assert out[0]["prediction_choice"] == "B"
    assert out[0]["correct"] is True
    assert out[0]["error"] == 0


# ---- NLL ----


def test_process_one_reuses_adversarial_forward_logits():
    class FakeRunner:
        def __init__(self):
            self.gradient_calls = 0
            self.original_calls = 0

        def prepare_inputs(self, _image, _question):
            return {"input_ids": torch.zeros(1, 2, dtype=torch.long)}

        def greedy_generate(self, _inputs):
            ids = torch.zeros(1, 1, dtype=torch.long)
            return {"text": "A", "answer_ids": ids, "answer_mask": torch.ones_like(ids).bool()}

        def build_teacher_forcing_inputs(self, _inputs, answer_ids, answer_mask):
            return {
                "answer_ids": answer_ids,
                "answer_mask": answer_mask,
                "prompt_length": 2,
                "answer_length": 1,
            }

        def compute_logprob_gradient(self, _teacher_inputs, _stage):
            self.gradient_calls += 1
            return {
                "gradient": torch.ones(1, 1, 2),
                "response_logits": torch.zeros(1, 1, 2),
            }

        def forward_original(self, _teacher_inputs):
            self.original_calls += 1
            return {"response_logits": torch.zeros(1, 1, 2)}

        def forward_with_perturbation(self, _teacher_inputs, _spec):
            return {"response_logits": torch.zeros(1, 1, 2)}

        @staticmethod
        def mean_nll(_logits, _ids, _mask):
            return 0.5

        @staticmethod
        def mean_kl(_original, _perturbed, _mask):
            return 0.0

    runner = FakeRunner()
    sample = {
        "id": "reuse",
        "dataset": "cvbench2d",
        "image": object(),
        "question": "question",
        "references": ["A"],
        "choices": ["yes", "no"],
        "answer_index": 0,
        "metadata": {},
    }
    process_one(
        sample,
        runner,
        modes=["adversarial"],
        stages=["fusion", "reasoning"],
        seeds=(42,),
        sigma=0.1,
        gamma=1.0,
        experiment_config={},
    )
    assert runner.gradient_calls == 2
    assert runner.original_calls == 0


def test_per_sample_nll_uniform_is_log2():
    # logits 全 0 → 2 类均匀 → p=0.5 → nll=ln2
    logits = torch.zeros(1, 1, 2)
    answer = torch.zeros(1, 1, dtype=torch.long)
    nll = LlavaMalpRunner.per_sample_nll(logits, answer)
    assert torch.allclose(nll, torch.tensor([math.log(2.0)]), atol=1e-6)


def test_per_sample_nll_mask_excludes_tokens():
    # 两个 token，mask 掩掉第二个；只算第一个（均匀→ln2）
    logits = torch.zeros(1, 2, 2)
    answer = torch.zeros(1, 2, dtype=torch.long)
    mask = torch.tensor([[True, False]])
    nll = LlavaMalpRunner.per_sample_nll(logits, answer, mask)
    assert torch.allclose(nll, torch.tensor([math.log(2.0)]), atol=1e-6)


def test_mean_nll_returns_float():
    logits = torch.zeros(1, 1, 2)
    answer = torch.zeros(1, 1, dtype=torch.long)
    value = LlavaMalpRunner.mean_nll(logits, answer)
    assert isinstance(value, float)
    assert math.isclose(value, math.log(2.0), abs_tol=1e-6)


def test_response_logits_uses_sequence_tail_after_image_expansion():
    # 3 prompt positions + 4 internally-expanded image positions + 2 answer
    # positions. Causal logits at positions 6 and 7 predict the answer tokens.
    logits = torch.arange(9, dtype=torch.float32).view(1, 9, 1)
    response = LlavaMalpRunner.response_logits(logits, prompt_length=3, answer_length=2)
    assert response.reshape(-1).tolist() == [6.0, 7.0]


def test_answer_mask_keeps_first_eos_and_masks_batch_padding():
    runner = object.__new__(LlavaMalpRunner)
    runner.eos_token_ids = (2,)
    runner.pad_token_id = 2
    answer_ids = torch.tensor([[10, 2, 2, 2], [11, 12, 2, 2]])
    mask = runner.build_answer_mask(answer_ids)
    assert mask.tolist() == [[True, True, False, False], [True, True, True, False]]


def test_question_mask_aligns_expanded_image_tokens():
    class FakeTokenizer:
        def __call__(self, _prompt, **_kwargs):
            return {
                "input_ids": torch.tensor([[10, 99, 20, 21, 30]]),
                "offset_mapping": torch.tensor(
                    [[[0, 1], [1, 8], [8, 11], [11, 13], [13, 14]]]
                ),
            }

    class FakeProcessor:
        tokenizer = FakeTokenizer()

    runner = object.__new__(LlavaMalpRunner)
    runner.processor = FakeProcessor()
    runner.image_token_index = 99
    input_ids = torch.tensor([[10, 99, 99, 99, 20, 21, 30]])
    attention_mask = torch.ones_like(input_ids)
    mask = runner._build_question_token_mask(
        "x<image>abcdey", "abcde", input_ids, attention_mask
    )
    assert mask.tolist() == [[False, False, False, False, True, True, False]]


def test_fusion_mask_selects_image_and_question_only():
    runner = object.__new__(LlavaMalpRunner)
    runner.image_token_index = 99
    teacher_inputs = {
        "input_ids": torch.tensor([[1, 99, 99, 20, 21, 30, 40, 41]]),
        "question_token_mask": torch.tensor([[False, False, False, True, True, False]]),
        "prompt_length": 6,
        "answer_length": 2,
    }
    mask = runner.build_fusion_mask(teacher_inputs)
    assert mask.tolist() == [[False, True, True, True, True, False, False, False]]


def test_reasoning_mask_matches_causal_answer_positions():
    teacher_inputs = {
        "attention_mask": torch.ones(1, 7, dtype=torch.long),
        "prompt_length": 4,
        "answer_length": 3,
        "answer_mask": torch.tensor([[True, True, False]]),
    }
    mask = LlavaMalpRunner.build_reasoning_mask(teacher_inputs)
    assert mask.tolist() == [[False, False, False, True, True, False, False]]


# ---- total_logprob ----


def test_total_logprob_uniform():
    logits = torch.zeros(1, 3, 2)
    answer = torch.zeros(1, 3, dtype=torch.long)
    total = LlavaMalpRunner.total_logprob(logits, answer)
    assert torch.allclose(total, torch.tensor(3 * math.log(0.5)), atol=1e-6)


# ---- KL ----


def test_kl_zero_for_identical_logits():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 5)
    kl = LlavaMalpRunner.kl_divergence_per_sample(logits, logits)
    assert torch.allclose(kl, torch.zeros(2), atol=1e-6)


def test_kl_non_negative():
    torch.manual_seed(1)
    original = torch.randn(2, 3, 5)
    perturbed = torch.randn(2, 3, 5)
    kl = LlavaMalpRunner.kl_divergence_per_sample(original, perturbed)
    assert (kl >= -1e-6).all()


def test_kl_matches_manual_value():
    # 单 token、2 类
    original = torch.tensor([[[0.0, 0.0]]])  # p = [0.5, 0.5]
    perturbed = torch.tensor([[[1.0, 0.0]]])  # q = softmax([1, 0])
    prob = [0.5, 0.5]
    q = [math.e / (math.e + 1.0), 1.0 / (math.e + 1.0)]
    expected = sum(p * (math.log(p) - math.log(qi)) for p, qi in zip(prob, q, strict=True))
    kl = LlavaMalpRunner.kl_divergence_per_sample(original, perturbed)
    assert torch.allclose(kl, torch.tensor([expected]), atol=1e-5)


def test_mean_kl_returns_float():
    logits = torch.randn(1, 2, 4)
    value = LlavaMalpRunner.mean_kl(logits, logits)
    assert isinstance(value, float)
    assert math.isclose(value, 0.0, abs_tol=1e-6)


# ---- evaluate_record / summarize 聚合 ----


def _perturbation(stage: str, mode: str, nll: float, kl: float, seed: int = 42) -> dict:
    return {
        "stage": stage,
        "modality": "joint",
        "mode": mode,
        "sigma": 0.1,
        "gamma": 1.0,
        "seed": seed,
        "nll": nll,
        "kl": kl,
    }


def test_evaluate_record_aggregates_pis_and_kl():
    record = {
        "id": "x",
        "dataset": "cvbench2d",
        "prediction": "The answer is (A)",
        "choices": ["A", "B", "C", "D"],
        "answer_index": 0,
        "nll0": 1.0,
        "perturbations": [
            _perturbation("fusion", "norm_isotropic", 1.5, 0.10, 42),
            _perturbation("fusion", "norm_isotropic", 1.7, 0.20, 43),
        ],
    }
    out = evaluate_record(record)
    assert len(out) == 1
    item = out[0]
    assert item["mode"] == "norm_isotropic"
    assert item["correct"] is True
    assert item["error"] == 0
    assert item["gold_choice"] == "A"
    assert item["prediction_choice"] == "A"
    assert item["stage"] == "fusion"
    assert math.isclose(item["pis"], 0.6)
    assert math.isclose(item["kl"], 0.15)


def test_evaluate_record_wrong_answer():
    record = {
        "id": "y",
        "prediction": "(B)",
        "choices": ["A", "B"],
        "answer_index": 0,
        "nll0": 0.5,
        "perturbations": [
            _perturbation("reasoning", "directional", 0.6, 0.01),
        ],
    }
    out = evaluate_record(record)
    assert out[0]["correct"] is False
    assert out[0]["error"] == 1


def test_evaluate_record_filter_modes():
    record = {
        "id": "z",
        "prediction": "(A)",
        "choices": ["A", "B"],
        "answer_index": 0,
        "nll0": 0.5,
        "perturbations": [
            _perturbation("fusion", "norm_isotropic", 0.6, 0.01),
            _perturbation("fusion", "directional", 0.7, 0.03),
        ],
    }
    out = evaluate_record(record, filter_modes={"directional"})
    assert {item["mode"] for item in out} == {"directional"}


def test_evaluate_record_rejects_duplicate_seed():
    record = {
        "id": "duplicate-seed",
        "prediction": "(A)",
        "choices": ["A", "B"],
        "answer_index": 0,
        "nll0": 0.5,
        "perturbations": [
            _perturbation("fusion", "norm_isotropic", 0.6, 0.01),
            _perturbation("fusion", "norm_isotropic", 0.7, 0.02),
        ],
    }
    try:
        evaluate_record(record)
    except ValueError as error:
        assert "missing/duplicate seeds" in str(error)
    else:
        raise AssertionError("duplicate seed should be rejected")


def test_summarize_computes_accuracy_and_auroc():
    expanded = [
        {
            "id": "a", "stage": "fusion", "mode": "norm_isotropic", "prediction": "A",
            "prediction_choice": "A", "gold_choice": "A", "correct": True, "error": 0,
            "nll0": 0.2, "nll_mean": 0.35, "pis": 0.15, "kl": 0.03,
            "layers": None,
        },
        {
            "id": "b", "stage": "fusion", "mode": "norm_isotropic", "prediction": "B",
            "prediction_choice": "B", "gold_choice": "A", "correct": False, "error": 1,
            "nll0": 0.8, "nll_mean": 1.4, "pis": 0.6, "kl": 0.15,
            "layers": None,
        },
    ]
    summaries = summarize(expanded)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["mode"] == "norm_isotropic"
    assert math.isclose(summary["accuracy"], 0.5)
    # 错误样本 nll0 更高 → nll0 预测 error 的 AUC=1
    assert math.isclose(summary["auroc_nll0_error"], 1.0)
    assert math.isclose(summary["auroc_pis_error"], 1.0)


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"ok: {name}")
    print("malp metrics tests passed")
