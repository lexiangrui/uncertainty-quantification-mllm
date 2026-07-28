import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import _cvbench_files, format_multiple_choice, format_question, iter_hallusionbench
from io_utils import append_jsonl, batches, load_jsonl_by_id
from perturb import (
    combined_uncertainty,
    add_norm_preserving_gaussian,
    nll_instability,
    replace_with_gaussian,
    select_sensitive_indices,
    semantic_volume,
    visual_dependency_scores,
)


def test_question_formatting():
    assert "(A) left" in format_multiple_choice("Where?", ["left", "right"])
    assert "Provide only the final answer" in format_question("Where?", None)


def test_modality_constrained_gradient_selection():
    scores = torch.tensor([100.0, 9.0, 8.0, 7.0, 99.0, 6.0])
    visual_mask = torch.tensor([False, True, True, True, False, False])
    text_mask = torch.tensor([False, False, False, False, True, True])
    assert select_sensitive_indices(scores, visual_mask, 0.5).tolist() == [1, 2]
    assert select_sensitive_indices(scores, text_mask, 0.5).tolist() == [4]


def test_gaussian_replacement_changes_only_selected_vectors():
    embeddings = torch.arange(30, dtype=torch.float32).view(1, 6, 5)
    selected = torch.tensor([1, 4])
    reference = torch.tensor([1, 2, 3, 4])
    changed_a = replace_with_gaussian(
        embeddings, selected, reference, seed=11, scale=1.0
    )
    changed_b = replace_with_gaussian(
        embeddings, selected, reference, seed=11, scale=1.0
    )
    untouched = torch.tensor([0, 2, 3, 5])
    assert torch.equal(changed_a[:, untouched], embeddings[:, untouched])
    assert not torch.equal(changed_a[:, selected], embeddings[:, selected])
    assert torch.equal(changed_a, changed_b)


def test_norm_isotropic_addition_preserves_selected_token_norms():
    embeddings = torch.randn(1, 6, 8, dtype=torch.float32)
    selected = torch.tensor([1, 4])
    changed = add_norm_preserving_gaussian(
        embeddings, selected, seed=11, sigma=0.5
    )
    untouched = torch.tensor([0, 2, 3, 5])
    assert torch.equal(changed[:, untouched], embeddings[:, untouched])
    assert not torch.equal(changed[:, selected], embeddings[:, selected])
    assert torch.allclose(
        changed[:, selected].norm(dim=-1),
        embeddings[:, selected].norm(dim=-1),
        atol=1e-6,
    )


def test_norm_isotropic_addition_is_reproducible():
    embeddings = torch.randn(1, 4, 8, dtype=torch.float32)
    selected = torch.tensor([0, 3])
    first = add_norm_preserving_gaussian(embeddings, selected, seed=23, sigma=0.1)
    second = add_norm_preserving_gaussian(embeddings, selected, seed=23, sigma=0.1)
    third = add_norm_preserving_gaussian(embeddings, selected, seed=37, sigma=0.1)
    assert torch.equal(first, second)
    assert not torch.equal(first[:, selected], third[:, selected])


def test_norm_isotropic_uses_raw_gaussian_before_norm_projection():
    embeddings = torch.tensor([[[3.0, 4.0]]], dtype=torch.float32)
    selected = torch.tensor([0])
    seed = 53
    sigma = 0.25
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(embeddings.shape, generator=generator)
    temporary = embeddings + sigma * noise
    expected = temporary * (
        embeddings.norm(dim=-1, keepdim=True) / temporary.norm(dim=-1, keepdim=True)
    )
    actual = add_norm_preserving_gaussian(
        embeddings, selected, seed=seed, sigma=sigma
    )
    assert torch.allclose(actual, expected, atol=1e-7)


def test_norm_isotropic_preserves_fp16_norm_with_rounding_tolerance():
    embeddings = torch.randn(1, 5, 32, dtype=torch.float16)
    selected = torch.tensor([1, 2, 4])
    changed = add_norm_preserving_gaussian(
        embeddings, selected, seed=71, sigma=0.01
    )
    assert torch.allclose(
        changed[:, selected].float().norm(dim=-1),
        embeddings[:, selected].float().norm(dim=-1),
        rtol=5e-4,
        atol=5e-4,
    )


def test_nll_instability():
    result = nll_instability(1.0, [1.0, 1.5, 0.5, 2.0, 1.0])
    assert 0.0 < result["score"] < 1.0
    assert result["mean_delta"] == 0.2
    assert result["mean_absolute_delta"] == 0.4


def test_semantic_volume():
    identical = [torch.tensor([1.0, 0.0])] * 5
    identical_result = semantic_volume(identical, jitter=1e-6)
    assert identical_result["score"] == 0.0
    assert len(identical_result["eigenvalues"]) == 5
    diverse = [
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
    ]
    diverse_result = semantic_volume(diverse, jitter=1e-6)
    assert diverse_result["score"] == 1.0
    assert diverse_result["log_volume"] > identical_result["log_volume"]
    assert len(diverse_result["cosine_gram_matrix"]) == 5


def test_visual_change_is_support_not_risk():
    weak_change = visual_dependency_scores(0.1, 0.2)
    strong_change = visual_dependency_scores(0.8, 0.9)
    assert strong_change["visual_dependency"] > weak_change["visual_dependency"]
    assert strong_change["visual_ungrounded_risk"] < weak_change[
        "visual_ungrounded_risk"
    ]


def test_combined_uncertainty_detects_both_failure_modes():
    low_confidence = combined_uncertainty(0.9, 0.1)
    high_confidence_ungrounded = combined_uncertainty(0.1, 0.9)
    confident_and_grounded = combined_uncertainty(0.1, 0.1)
    assert low_confidence > confident_and_grounded
    assert high_confidence_ungrounded > confident_and_grounded
    assert combined_uncertainty(0.2, 0.2) > combined_uncertainty(0.2, 0.1)


def test_stronger_visual_change_monotonically_lowers_uncertainty():
    weak = visual_dependency_scores(0.1, 0.2)
    strong = visual_dependency_scores(0.8, 0.9)
    assert combined_uncertainty(0.2, strong["visual_ungrounded_risk"]) < combined_uncertainty(
        0.2, weak["visual_ungrounded_risk"]
    )


def test_inference_tensor_can_be_cloned_for_grad_work():
    with torch.inference_mode():
        inference_target = torch.tensor([1], dtype=torch.long)
    target = inference_target.clone()
    logits = torch.randn(1, 2, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(logits, target)
    loss.backward()
    assert logits.grad is not None


def test_cvbench_subset_path_selection(tmp_path):
    base = tmp_path / "nyu-visionx___cv-bench"
    two_d = base / "2D" / "0.0.0" / "hash" / "cv-bench-test.arrow"
    three_d = base / "3D" / "0.0.0" / "hash" / "cv-bench-test.arrow"
    two_d.parent.mkdir(parents=True)
    three_d.parent.mkdir(parents=True)
    two_d.touch()
    three_d.touch()
    assert _cvbench_files(tmp_path, "2D") == [two_d]
    assert _cvbench_files(tmp_path) == [two_d, three_d]


def test_jsonl_checkpoint_resume(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        append_jsonl(handle, {"id": "a", "value": 1})
        append_jsonl(handle, {"id": "b", "value": 2})
    assert load_jsonl_by_id(path) == {
        "a": {"id": "a", "value": 1},
        "b": {"id": "b", "value": 2},
    }


def test_batch_partitioning():
    items = [{"id": str(index)} for index in range(19)]
    groups = list(batches(items, 8))
    assert [len(group) for group in groups] == [8, 8, 3]


def test_hallusionbench_image_adapter(tmp_path, monkeypatch):
    import pandas as pd

    path = tmp_path / "HallusionBench" / "data" / "image-00000-of-00001.parquet"
    path.parent.mkdir(parents=True)
    path.touch()
    row = {
        "image": {"bytes": b"image-bytes"},
        "gt_answer": "1",
        "question": "Is there a cat?",
        "category": "VD",
        "subcategory": "animal",
        "visual_input": "1",
        "set_id": "0",
        "figure_id": "0",
        "question_id": "0",
        "sample_note": "",
        "filename": "sample.png",
        "gt_answer_details": "A cat is visible.",
    }
    monkeypatch.setattr(pd, "read_parquet", lambda _path: pd.DataFrame([row]))

    class FakeImage:
        def convert(self, mode):
            assert mode == "RGB"
            return self

    monkeypatch.setattr("data.Image.open", lambda _stream: FakeImage())
    sample = next(iter_hallusionbench(tmp_path, limit=1))
    assert sample["answer_index"] == 1
    assert sample["judge_mode"] == "yes_no"
    assert sample["choices"] == ["No", "Yes"]
