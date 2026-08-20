from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation import run_metrics


GREEDY_RUN = {"dataset": "vilp", "model_family": "llava_1_5", "model_id": "llava"}

UQ_RUN = {
    "greedy_run": GREEDY_RUN,
    "uq_methods": [{"name": "perplexity"}, {"name": "semantic_entropy"}],
}

JUDGE_RUN = {"judge_model": "judge", "greedy_run": GREEDY_RUN}

EVALUATED = [
    # sample_id, group_id, correct, perplexity score
    ("s1", "g1", True, 1.0),
    ("s2", "g1", True, 1.5),
    ("s3", "g2", False, 5.0),
    ("s4", "g2", True, 2.0),
    ("s5", "g3", False, 6.0),
    ("s6", "g3", True, 2.5),
    ("s7", "g4", False, 7.0),
    ("s8", "g4", True, 3.0),
]


def _judge_record(sample_id: str, group_id: str, *, valid: bool, correct: bool) -> dict:
    if valid:
        judge = {
            "valid": True,
            "error": None,
            "correct": correct,
            "rating": 4,
            "hallucination": False,
            "hallucination_types": [],
        }
    else:
        judge = {
            "valid": False,
            "error": "judge response is not valid JSON",
            "correct": None,
            "rating": None,
            "hallucination": None,
            "hallucination_types": None,
        }
    return {
        "record_type": "sample",
        "sample": {
            "sample_id": sample_id,
            "group_id": group_id,
            "dataset": "vilp",
            "split": "test",
        },
        "judge": judge,
    }


def _uq_record(sample_id: str, perplexity: float | None, entropy: float = 0.5) -> dict:
    if perplexity is None:
        perplexity_value = {"valid": False, "error": "missing signals", "score": None}
    else:
        perplexity_value = {"valid": True, "error": None, "score": perplexity}
    return {
        "record_type": "sample",
        "sample": {"sample_id": sample_id},
        "uq": {
            "perplexity": perplexity_value,
            "semantic_entropy": {"valid": True, "error": None, "score": entropy},
        },
    }


def _write_jsonl(path: Path, run: dict, records: list[dict]) -> Path:
    header = {"record_type": "run", "run": run}
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [header, *records])
    )
    return path


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    judge_records = [
        _judge_record(sample_id, group_id, valid=True, correct=correct)
        for sample_id, group_id, correct, _ in EVALUATED
    ]
    judge_records.append(_judge_record("s9", "g5", valid=False, correct=False))
    judge_records.append(_judge_record("s10", "g5", valid=True, correct=True))
    judge_records.append(_judge_record("s11", "g6", valid=True, correct=False))
    uq_records = [
        _uq_record(sample_id, perplexity)
        for sample_id, _, _, perplexity in EVALUATED
    ]
    uq_records.append(_uq_record("s11", None))
    uq_records.append(_uq_record("s12", 4.0))
    judge_input = _write_jsonl(tmp_path / "judge.jsonl", JUDGE_RUN, judge_records)
    uq_input = _write_jsonl(tmp_path / "uq.jsonl", UQ_RUN, uq_records)
    return uq_input, judge_input


def test_run_metrics_reports_counts_labels_and_both_targets(tmp_path: Path) -> None:
    uq_input, judge_input = _write_inputs(tmp_path)
    output = tmp_path / "metrics.json"
    report = run_metrics(
        uq_input=uq_input,
        judge_input=judge_input,
        output=output,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )

    assert report["dataset"] == "vilp"
    assert report["model_family"] == "llava_1_5"
    assert report["uq_methods"] == ["perplexity", "semantic_entropy"]
    assert report["counts"] == {
        "judge_records": 11,
        "uq_records": 10,
            "evaluated": 10,
            "clusters": 6,
        "excluded": {
            "invalid_judge": 1,
            "missing_uq_record": 1,
            "invalid_uq_score": 1,
            "invalid_uq_score_by_method": {"perplexity": 1, "semantic_entropy": 0},
            "uq_without_judge_record": 1,
        },
    }

    labels = report["labels"]
    assert labels["accuracy"]["value"] == pytest.approx(0.6)
    assert labels["accuracy"]["ci_low"] <= 0.6 <= labels["accuracy"]["ci_high"]
    assert labels["hallucination_rate"]["value"] == pytest.approx(0.0)
    assert labels["hallucination_rate"]["ci_low"] == pytest.approx(0.0)
    assert labels["joint_counts"] == {
        "correct_without_hallucination": 6,
        "correct_with_hallucination": 0,
        "wrong_without_hallucination": 4,
        "wrong_with_hallucination": 0,
    }

    error_target = report["targets"]["error"]
    assert error_target["positives"] == 4
    assert error_target["positive_rate"] == pytest.approx(0.4)
    perplexity = error_target["methods"]["perplexity"]
    assert set(perplexity) == {"auroc", "auprc", "prr", "n", "positives"}
    assert perplexity["n"] == 8
    assert perplexity["positives"] == 3
    assert perplexity["auroc"]["value"] == pytest.approx(1.0)
    assert perplexity["auprc"]["value"] == pytest.approx(1.0)
    assert perplexity["prr"]["value"] == pytest.approx(1.0)
    assert perplexity["auroc"]["ci_low"] == pytest.approx(1.0)
    assert perplexity["auroc"]["ci_high"] == pytest.approx(1.0)
    entropy = error_target["methods"]["semantic_entropy"]
    assert entropy["n"] == 9
    assert entropy["positives"] == 4
    assert entropy["auroc"]["value"] == pytest.approx(0.5)
    assert entropy["auprc"]["value"] == pytest.approx(4 / 9)
    assert entropy["prr"]["value"] == pytest.approx(0.0)

    hallucination_target = report["targets"]["hallucination"]
    assert hallucination_target["positives"] == 0
    for method in hallucination_target["methods"].values():
        for name in ("auroc", "auprc", "prr"):
            assert method[name]["value"] is None
            assert method[name]["reason"] == "target labels contain a single class"

    written = json.loads(output.read_text())
    assert written == report


def test_run_metrics_is_deterministic(tmp_path: Path) -> None:
    uq_input, judge_input = _write_inputs(tmp_path)
    first = run_metrics(
        uq_input=uq_input,
        judge_input=judge_input,
        output=tmp_path / "first.json",
        bootstrap_samples=50,
        bootstrap_seed=3,
    )
    second = run_metrics(
        uq_input=uq_input,
        judge_input=judge_input,
        output=tmp_path / "second.json",
        bootstrap_samples=50,
        bootstrap_seed=3,
    )
    assert first == second


def test_run_metrics_rejects_greedy_run_mismatch(tmp_path: Path) -> None:
    uq_input, _ = _write_inputs(tmp_path)
    other_run = {
        "judge_model": "judge",
        "greedy_run": {**GREEDY_RUN, "model_id": "other"},
    }
    judge_input = _write_jsonl(
        tmp_path / "judge-mismatch.jsonl",
        other_run,
        [_judge_record("s1", "g1", valid=True, correct=True)],
    )
    with pytest.raises(ValueError, match="greedy_run mismatch"):
        run_metrics(
            uq_input=uq_input,
            judge_input=judge_input,
            output=tmp_path / "metrics.json",
        )


def test_run_metrics_rejects_missing_run_header(tmp_path: Path) -> None:
    uq_input, judge_input = _write_inputs(tmp_path)
    headerless = tmp_path / "headerless.jsonl"
    headerless.write_text(
        json.dumps(_judge_record("s1", "g1", valid=True, correct=True)) + "\n"
    )
    with pytest.raises(ValueError, match="missing run header"):
        run_metrics(
            uq_input=uq_input,
            judge_input=headerless,
            output=tmp_path / "metrics.json",
        )


def test_run_metrics_requires_evaluable_samples(tmp_path: Path) -> None:
    judge_input = _write_jsonl(
        tmp_path / "judge.jsonl",
        JUDGE_RUN,
        [_judge_record("s1", "g1", valid=False, correct=False)],
    )
    uq_input = _write_jsonl(tmp_path / "uq.jsonl", UQ_RUN, [_uq_record("s1", 1.0)])
    with pytest.raises(ValueError, match="no samples left"):
        run_metrics(
            uq_input=uq_input,
            judge_input=judge_input,
            output=tmp_path / "metrics.json",
        )
