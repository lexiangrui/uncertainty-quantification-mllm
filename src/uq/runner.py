from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Protocol

import torch

from src.generation.runner import ResponseSignals
from src.generation.records import FORMAT_SKIP_POLICY, has_valid_response_format
from src.utils import completed_sample_ids, load_jsonl_records, write_sample_json_line


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class UQMethod(Protocol):
    required_responses: str

    @property
    def runtime_config(self) -> dict: ...

    def compute(
        self,
        *,
        question: str,
        greedy: ResponseSignals,
        samples: list[ResponseSignals],
    ) -> dict: ...


def _load_generation(path: Path) -> tuple[dict, list[dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = load_jsonl_records(path)
    if not rows or rows[0].get("record_type") != "run":
        raise ValueError(f"generation input lacks a run header: {path}")
    records = rows[1:]
    if any(record.get("record_type") != "sample" for record in records):
        raise ValueError(f"generation input contains an invalid record: {path}")
    return rows[0]["run"], records


def _signal(record: dict, hidden: tuple[float, ...] = ()) -> ResponseSignals | None:
    values = record.get("signals")
    answer = record.get("uq_text", record.get("answer"))
    if not isinstance(values, dict) or not isinstance(answer, str):
        return None
    return ResponseSignals(
        answer=answer,
        token_count=int(values["token_count"]),
        sequence_log_prob=float(values["sequence_log_prob"]),
        mean_log_prob=float(values["mean_log_prob"]),
        sampling_sequence_log_prob=float(values["sampling_sequence_log_prob"]),
        final_hidden=hidden,
    )


def _load_sample_signals(sample_input: Path, record: dict) -> list[ResponseSignals | None]:
    descriptor = record.get("hidden_states")
    samples = record.get("samples", [])
    if descriptor is None:
        return [_signal(sample) for sample in samples]
    relative = Path(descriptor["path"])
    if relative.is_absolute():
        raise ValueError("hidden-state path must be relative")
    storage = descriptor.get("storage", "generation_adjacent")
    if storage == "results_hidden":
        root = (PROJECT_ROOT / "results" / "hidden").resolve()
    elif storage == "generation_adjacent":
        root = sample_input.parent.resolve()
    else:
        raise ValueError(f"unknown hidden-state storage: {storage}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("hidden-state path escapes its storage root")
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError(f"invalid hidden-state tensor: {path}")
    if list(tensor.shape) != descriptor.get("shape"):
        raise ValueError(f"hidden-state shape mismatch: {path}")
    values: list[ResponseSignals | None] = []
    for position, sample in enumerate(samples):
        index = sample.get("hidden_state_index", position)
        if not isinstance(index, int) or not 0 <= index < tensor.shape[0]:
            raise ValueError(f"invalid hidden_state_index in {path}")
        values.append(_signal(sample, tuple(tensor[index].float().tolist())))
    return values


def _finalize(output: Path, methods: tuple[UQMethod, ...]) -> None:
    finalizers = [method for method in methods if hasattr(method, "finalize")]
    if not finalizers or not output.exists():
        return
    with output.open(encoding="utf-8") as handle:
        rows = load_jsonl_records(output)
    for method in finalizers:
        name = method.runtime_config["name"]
        method.finalize([record["uq"][name] for record in rows[1:]])
    descriptor, temporary = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def run_split_uq(
    *,
    greedy_input: Path,
    sample_input: Path,
    output: Path,
    methods: tuple[UQMethod, ...],
) -> tuple[int, int]:
    """Compute UQ from separately frozen greedy and sampled-response runs."""
    greedy_run, greedy_records = _load_generation(greedy_input)
    sample_run, sample_rows = _load_generation(sample_input)
    if greedy_run.get("dataset") != sample_run.get("dataset"):
        raise ValueError("greedy and sample inputs use different datasets")
    if greedy_run.get("model_family") != sample_run.get("model_family"):
        raise ValueError("greedy and sample inputs use different model families")
    greedy_by_id = {
        record.get("sample", {}).get("sample_id"): record for record in greedy_records
    }
    if None in greedy_by_id or len(greedy_by_id) != len(greedy_records):
        raise ValueError("greedy input has invalid or duplicate sample IDs")
    run = {
        "uq_output_version": "split-uq-v1",
        "greedy_input": str(greedy_input.resolve()),
        "greedy_run": greedy_run,
        "sample_input": str(sample_input.resolve()),
        "sample_run": sample_run,
        "uq_methods": [method.runtime_config for method in methods],
        "invalid_format_policy": FORMAT_SKIP_POLICY,
    }
    completed = completed_sample_ids(output, run)
    written = skipped = 0
    for sample_record in sample_rows:
        sample = sample_record.get("sample", {})
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str):
            raise ValueError("sample input lacks sample_id")
        if sample_id in completed:
            skipped += 1
            continue
        greedy_record = greedy_by_id.get(sample_id)
        if greedy_record is None:
            raise ValueError(f"greedy input lacks sample_id: {sample_id}")
        greedy_value = greedy_record.get("greedy", {})
        sample_values = sample_record.get("samples", [])
        if greedy_value.get("sections_valid") is not True or not isinstance(sample_values, list) or not sample_values or not all(
            value.get("sections_valid") is True for value in sample_values
        ):
            skipped += 1
            continue
        greedy = _signal(greedy_value)
        samples = _load_sample_signals(sample_input, sample_record)
        uq = {}
        for method in methods:
            missing = None
            if method.required_responses in {"greedy", "greedy_and_samples"} and greedy is None:
                missing = "greedy response lacks valid generation signals"
            if method.required_responses in {"samples", "greedy_and_samples"} and any(
                value is None for value in samples
            ):
                missing = "one or more sampled responses lack valid generation signals"
            name = method.runtime_config["name"]
            if missing is not None:
                uq[name] = {"valid": False, "error": missing, "score": None}
            else:
                try:
                    uq[name] = method.compute(
                        question=sample["question"],
                        greedy=greedy,
                        samples=[value for value in samples if value is not None],
                    )
                except (ValueError, FloatingPointError, RuntimeError) as exc:
                    uq[name] = {"valid": False, "error": str(exc), "score": None}
        write_sample_json_line(output, run, {"sample": {"sample_id": sample_id}, "uq": uq})
        written += 1
    _finalize(output, methods)
    return written, skipped
