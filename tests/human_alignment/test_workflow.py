from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.datasets import BenchmarkSample
import src.human_alignment.workflow as workflow
from src.human_alignment.workflow import (
    _load_judge_file,
    build_alignment_workspace,
    finalize_aligned_results,
    load_annotations,
    save_annotations,
)


def _write_judge(path: Path, judge_model: str, labels: list[tuple[str, bool, bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run = {
        "judge_model": judge_model,
        "dataset": "vilp",
        "dataset_source": "/unused/dataset",
        "greedy_input": "/unused/greedy.jsonl",
        "greedy_run": {"dataset": "vilp", "model_family": "test"},
    }
    rows = [{"record_type": "run", "run": run}]
    for sample_id, correct, hallucination in labels:
        rows.append(
            {
                "record_type": "sample",
                "sample": {"sample_id": sample_id, "group_id": f"g-{sample_id}", "dataset": "vilp", "split": "test"},
                "input": {
                    "question": f"question {sample_id}",
                    "references": ["reference"],
                    "vision": "vision",
                    "reasoning": "reasoning",
                    "answer": "answer",
                    "raw_response": "raw",
                },
                "judge": {
                    "status": "ok",
                    "valid": True,
                    "error": None,
                    "analysis": judge_model,
                    "correct": correct,
                    "rating": 1 if hallucination else 5,
                    "hallucination": hallucination,
                    "hallucination_types": ["vision_hallucination"] if hallucination else [],
                    "raw_response": "judge raw",
                },
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    gpt = tmp_path / "judging_gpt"
    gemini = tmp_path / "judging_gemini"
    workspace = tmp_path / "human_alignment"
    _write_judge(
        gpt / "llava/vilp.jsonl",
        "gpt-5.6-terra",
        [("agree", True, False), ("correct", True, False), ("hallucination", False, True), ("both", True, True)],
    )
    _write_judge(
        gemini / "llava/vilp.jsonl",
        "gemini-3.7-flash",
        [("agree", True, False), ("correct", False, False), ("hallucination", False, False), ("both", False, False)],
    )
    return gpt, gemini, workspace


def test_prepare_splits_disagreement_dimensions_and_preserves_annotations(tmp_path: Path) -> None:
    gpt, gemini, workspace = _inputs(tmp_path)
    result = build_alignment_workspace(
        gpt_dir=gpt,
        gemini_dir=gemini,
        workspace=workspace,
        models=("llava",),
        datasets=("vilp",),
        export_images=False,
    )
    assert result["counts"] == {"correct": 2, "hallucination": 2, "overlap": 1, "unique": 3}
    assert {row["sample_id"] for row in result["samples"]} == {"correct", "hallucination", "both"}
    assert all("judge" not in row for row in result["samples"])

    save_annotations(workspace, {"llava/vilp/correct": {"correct": False, "annotator": "A"}})
    build_alignment_workspace(
        gpt_dir=gpt,
        gemini_dir=gemini,
        workspace=workspace,
        models=("llava",),
        datasets=("vilp",),
        export_images=False,
    )
    assert load_annotations(workspace)["annotations"]["llava/vilp/correct"]["correct"] is False


def test_save_annotations_rejects_nonhuman_provenance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid correct provenance"):
        save_annotations(
            tmp_path,
            {
                "llava/vilp/sample": {
                    "correct": True,
                    "provenance": {
                        "correct": {"kind": "automated", "model": "third-judge"}
                    },
                }
            },
        )


def test_finalize_refuses_incomplete_then_writes_human_aligned_labels(tmp_path: Path) -> None:
    gpt, gemini, workspace = _inputs(tmp_path)
    build_alignment_workspace(
        gpt_dir=gpt,
        gemini_dir=gemini,
        workspace=workspace,
        models=("llava",),
        datasets=("vilp",),
        export_images=False,
    )
    output = tmp_path / "judging"
    with pytest.raises(ValueError, match="incomplete"):
        finalize_aligned_results(
            gpt_dir=gpt,
            gemini_dir=gemini,
            workspace=workspace,
            output_dir=output,
            models=("llava",),
            datasets=("vilp",),
        )
    save_annotations(
        workspace,
        {
            "llava/vilp/correct": {"correct": False},
            "llava/vilp/hallucination": {"hallucination": False},
            "llava/vilp/both": {"correct": True, "hallucination": True, "hallucination_types": ["reasoning_hallucination"]},
        },
    )
    manifest = finalize_aligned_results(
        gpt_dir=gpt,
        gemini_dir=gemini,
        workspace=workspace,
        output_dir=output,
        models=("llava",),
        datasets=("vilp",),
    )
    assert manifest["files"][0]["adjudicated"] == 3
    rows = [json.loads(line) for line in (output / "llava/vilp.jsonl").read_text().splitlines()]
    assert rows[0]["run"]["protocol"] == "human-aligned-dual-judge-v1"
    labels = {row["sample"]["sample_id"]: row["judge"] for row in rows[1:]}
    assert (labels["agree"]["correct"], labels["agree"]["hallucination"]) == (True, False)
    assert labels["correct"]["correct"] is False
    assert labels["hallucination"]["hallucination"] is False
    assert labels["both"]["hallucination_types"] == ["reasoning_hallucination"]
    assert labels["both"]["alignment"]["correct"]["kind"] == "human"
    assert labels["correct"]["alignment"]["correct"] == {"kind": "human", "model": None}
    assert labels["hallucination"]["alignment"]["hallucination"] == {"kind": "human", "model": None}


def test_finalize_detects_raw_input_change(tmp_path: Path) -> None:
    gpt, gemini, workspace = _inputs(tmp_path)
    build_alignment_workspace(
        gpt_dir=gpt,
        gemini_dir=gemini,
        workspace=workspace,
        models=("llava",),
        datasets=("vilp",),
        export_images=False,
    )
    path = gpt / "llava/vilp.jsonl"
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw judge input changed"):
        finalize_aligned_results(
            gpt_dir=gpt,
            gemini_dir=gemini,
            workspace=workspace,
            output_dir=tmp_path / "judging",
            models=("llava",),
            datasets=("vilp",),
        )


def test_prepare_rejects_wrong_judge_identity(tmp_path: Path) -> None:
    gpt, gemini, workspace = _inputs(tmp_path)
    path = gemini / "llava/vilp.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["run"]["judge_model"] = "gpt-5.6-terra"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected Gemini judge_model"):
        build_alignment_workspace(
            gpt_dir=gpt,
            gemini_dir=gemini,
            workspace=workspace,
            models=("llava",),
            datasets=("vilp",),
            export_images=False,
        )


def test_loader_ignores_failed_attempt_before_valid_retry(tmp_path: Path) -> None:
    path = tmp_path / "judge.jsonl"
    _write_judge(path, "gpt-5.6-terra", [("sample", True, False)])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    failed = {
        "record_type": "sample",
        "sample": rows[1]["sample"],
        "input": rows[1]["input"],
        "judge": {"status": "api_error", "valid": False},
    }
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in [rows[0], failed, rows[1]]),
        encoding="utf-8",
    )
    _run, records = _load_judge_file(path)
    assert records["sample"]["judge"]["valid"] is True


def test_image_export_distinguishes_available_and_no_image(tmp_path: Path, monkeypatch) -> None:
    samples = [
        {"dataset": "vilp", "sample_id": "with-image"},
        {"dataset": "vilp", "sample_id": "without-image"},
    ]

    def fake_iter_dataset(_dataset, _source):
        yield BenchmarkSample("with-image", "g1", "vilp", "test", "q", ("a",), Image.new("RGB", (4, 4)))
        yield BenchmarkSample("without-image", "g2", "vilp", "test", "q", ("a",), None)

    monkeypatch.setattr(workflow, "iter_dataset", fake_iter_dataset)
    workflow._export_images(samples, {"vilp": "/unused"}, tmp_path)
    assert samples[0]["image_status"] == "available"
    assert (tmp_path / samples[0]["image"]).is_file()
    assert samples[1]["image_status"] == "not_applicable"
    assert samples[1]["image"] is None


def test_image_export_fails_when_alignment_sample_is_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(workflow, "iter_dataset", lambda _dataset, _source: iter(()))
    with pytest.raises(ValueError, match="not found in dataset"):
        workflow._export_images(
            [{"dataset": "vilp", "sample_id": "missing"}],
            {"vilp": "/unused"},
            tmp_path,
        )
