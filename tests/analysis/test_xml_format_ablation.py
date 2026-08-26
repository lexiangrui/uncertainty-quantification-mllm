from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.ablation.xml_format import (
    DATASETS,
    MANIFEST_PROTOCOL,
    complete_xml_sampling_frame,
    materialize_generation_subset,
    materialize_existing_judge_subset,
    summarize_model,
)


def _jsonl(path: Path, run: dict, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [{"record_type": "run", "run": run}, *rows]
    path.write_text("\n".join(json.dumps(value) for value in values) + "\n")


def _manifest() -> dict:
    samples = [
        {"sample_id": f"{dataset}-1", "group_id": f"group-{dataset}", "dataset": dataset}
        for dataset in DATASETS
    ]
    canonical = json.dumps(samples, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "protocol": MANIFEST_PROTOCOL,
        "sample_size": len(samples),
        "sample_set_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "samples": samples,
    }


def _generation_row(dataset: str, valid: bool = True) -> dict:
    return {
        "record_type": "sample",
        "sample": {"sample_id": f"{dataset}-1", "group_id": f"group-{dataset}"},
        "greedy": {"sections_valid": valid},
    }


def _judge_row(dataset: str, *, correct: bool, hallucination: bool) -> dict:
    return {
        "record_type": "sample",
        "sample": {"sample_id": f"{dataset}-1", "group_id": f"group-{dataset}"},
        "judge": {
            "valid": True,
            "correct": correct,
            "hallucination": hallucination,
        },
    }


def test_materialize_generation_subset_preserves_only_selected_records(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    selected = _generation_row("vilp")
    other = {
        "record_type": "sample",
        "sample": {"sample_id": "vilp-2", "group_id": "group-2"},
        "greedy": {"sections_valid": True},
    }
    _jsonl(source, {"prompt_sha256": "original"}, [selected, other])
    output = tmp_path / "subset.jsonl"
    assert materialize_generation_subset(
        source=source,
        output=output,
        sample_ids={"vilp-1"},
        manifest=_manifest(),
    ) == 1
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["run"]["ablation_subset"]["condition"] == "xml_lora"
    assert [row["sample"]["sample_id"] for row in rows[1:]] == ["vilp-1"]


def test_complete_xml_sampling_frame_intersects_all_models(tmp_path: Path) -> None:
    root = tmp_path / "generation"
    for model in ("llava", "qwen", "internvl"):
        for dataset in DATASETS:
            rows = [_generation_row(dataset)]
            if model == "llava" and dataset == "vilp":
                rows.append(
                    {
                        "record_type": "sample",
                        "sample": {"sample_id": "vilp-incomplete", "group_id": "g"},
                        "greedy": {"sections_valid": False},
                    }
                )
            _jsonl(root / model / "greedy" / f"{dataset}.jsonl", {}, rows)
    eligible, counts = complete_xml_sampling_frame(root)
    assert eligible["vilp"] == {"vilp-1"}
    assert "vilp-incomplete" not in eligible["vilp"]
    assert counts["vilp"]["shared_complete"] == 1


def test_materialize_existing_judge_subset_reuses_gemini_label(tmp_path: Path) -> None:
    source = tmp_path / "gemini.jsonl"
    selected = _judge_row("vilp", correct=True, hallucination=False)
    other = {
        "record_type": "sample",
        "sample": {"sample_id": "vilp-2", "group_id": "group-2"},
        "judge": {"valid": True, "correct": False, "hallucination": True},
    }
    _jsonl(
        source,
        {"protocol": "openai-responses", "judge_model": "gemini-3.7-flash"},
        [selected, other],
    )
    output = tmp_path / "labels.jsonl"
    assert materialize_existing_judge_subset(
        source=source,
        output=output,
        sample_ids={"vilp-1"},
        manifest=_manifest(),
        judge_model="gemini-3.7-flash",
    ) == 1
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    subset = rows[0]["run"]["ablation_subset"]
    assert subset["condition"] == "xml_lora"
    assert subset["label_source"] == "existing_single_gemini_judge_labels"
    assert rows[1]["judge"]["correct"] is True


def test_summarize_model_uses_paired_valid_records(tmp_path: Path) -> None:
    manifest = _manifest()
    generation_root = tmp_path / "generation"
    judge_root = tmp_path / "judge"
    labels = {
        "vilp": (True, False, False, True),
        "hallusionbench": (True, False, True, False),
        "mmvet": (False, True, False, True),
    }
    for condition in ("xml_lora", "native_prompt"):
        for dataset in DATASETS:
            sections_valid = not (
                condition == "native_prompt" and dataset == "mmvet"
            )
            _jsonl(
                generation_root / condition / "llava" / f"{dataset}.jsonl",
                {"condition": condition},
                [_generation_row(dataset, valid=sections_valid)],
            )
            xml_correct, xml_hall, native_correct, native_hall = labels[dataset]
            correct, hallucination = (
                (xml_correct, xml_hall)
                if condition == "xml_lora"
                else (native_correct, native_hall)
            )
            _jsonl(
                judge_root / condition / "llava" / f"{dataset}.jsonl",
                {"condition": condition},
                [_judge_row(dataset, correct=correct, hallucination=hallucination)],
            )
    format_rows, metric_rows = summarize_model(
        model="llava",
        manifest=manifest,
        generation_root=generation_root,
        judge_root=judge_root,
        bootstrap_samples=50,
        bootstrap_seed=0,
    )
    assert next(
        row for row in format_rows if row["condition"] == "xml_lora"
    )["format_valid_n"] == 3
    assert next(
        row for row in format_rows if row["condition"] == "native_prompt"
    )["format_valid_n"] == 2
    accuracy = next(row for row in metric_rows if row["metric"] == "accuracy")
    hallucination = next(row for row in metric_rows if row["metric"] == "hallucination_rate")
    assert accuracy["candidate_n"] == 3
    assert accuracy["native_three_part_n"] == 2
    assert accuracy["paired_n"] == 2
    assert accuracy["xml_lora_rate"] == pytest.approx(1.0)
    assert accuracy["native_prompt_rate"] == pytest.approx(0.5)
    assert hallucination["xml_lora_rate"] == pytest.approx(0.0)
    assert hallucination["native_prompt_rate"] == pytest.approx(0.5)
