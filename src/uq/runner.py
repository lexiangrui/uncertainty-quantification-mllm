from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Protocol

import torch

from src.generation.runner import ResponseSignals
from src.generation.records import FORMAT_SKIP_POLICY, has_valid_response_format
from src.utils import completed_sample_ids, write_sample_json_line


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
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or rows[0].get("record_type") != "run":
        raise ValueError(f"generation input lacks a run header: {path}")
    records = rows[1:]
    if any(record.get("record_type") != "sample" for record in records):
        raise ValueError(f"generation input contains an invalid record: {path}")
    return rows[0]["run"], records


def _signal(record: dict, hidden: tuple[float, ...] = ()) -> ResponseSignals | None:
    values = record.get("signals")
    answer = record.get("answer")
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


def _load_hidden_manifest(path: Path) -> tuple[dict, dict[str, dict]]:
    run, records = _load_generation(path)
    values: dict[str, dict] = {}
    for record in records:
        sample_id = record.get("sample", {}).get("sample_id")
        descriptor = record.get("hidden_states")
        if not isinstance(sample_id, str) or not isinstance(descriptor, dict):
            raise ValueError(f"invalid hidden-state manifest record: {path}")
        if sample_id in values:
            raise ValueError(f"duplicate hidden-state sample_id in {path}: {sample_id}")
        values[sample_id] = descriptor
    return run, values


def _load_sample_signals(
    generation_input: Path,
    record: dict,
    *,
    hidden_root: Path | None = None,
    hidden_descriptor: dict | None = None,
) -> list[ResponseSignals | None]:
    external_hidden = hidden_descriptor is not None
    descriptor = hidden_descriptor or record.get("hidden_states")
    samples = record.get("samples", [])
    if descriptor is None:
        return [_signal(sample) for sample in samples]
    relative = Path(descriptor["path"])
    if relative.is_absolute():
        raise ValueError("hidden-state path must be relative")
    root = (hidden_root or generation_input.parent).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("hidden-state path escapes the generation directory")
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise ValueError(f"invalid hidden-state tensor: {path}")
    if list(tensor.shape) != descriptor.get("shape"):
        raise ValueError(f"hidden-state shape mismatch: {path}")
    values: list[ResponseSignals | None] = []
    for position, sample in enumerate(samples):
        index = sample.get("hidden_state_index", position if external_hidden else None)
        if not isinstance(index, int) or not 0 <= index < tensor.shape[0]:
            raise ValueError(f"invalid hidden_state_index in {path}")
        values.append(_signal(sample, tuple(tensor[index].float().tolist())))
    return values


def _finalize(output: Path, methods: tuple[UQMethod, ...]) -> None:
    finalizers = [method for method in methods if hasattr(method, "finalize")]
    if not finalizers or not output.exists():
        return
    with output.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
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


def run_deferred_uq(
    *,
    generation_input: Path,
    output: Path,
    methods: tuple[UQMethod, ...],
    hidden_input: Path | None = None,
) -> tuple[int, int]:
    generation_run, generation_records = _load_generation(generation_input)
    hidden_run = None
    hidden_records: dict[str, dict] = {}
    if hidden_input is not None:
        hidden_run, hidden_records = _load_hidden_manifest(hidden_input)
    run = {
        "uq_output_version": "deferred-uq-v1",
        "generation_input": str(generation_input.resolve()),
        "generation_run": generation_run,
        "hidden_input": str(hidden_input.resolve()) if hidden_input else None,
        "hidden_run": hidden_run,
        "uq_methods": [method.runtime_config for method in methods],
        "invalid_format_policy": FORMAT_SKIP_POLICY,
    }
    completed = completed_sample_ids(output, run)
    written = 0
    skipped = 0
    for record in generation_records:
        sample = record.get("sample", {})
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str):
            raise ValueError("generation record lacks sample_id")
        if sample_id in completed:
            skipped += 1
            continue
        if not has_valid_response_format(record):
            skipped += 1
            continue
        if hidden_input is not None and sample_id not in hidden_records:
            raise ValueError(f"hidden-state manifest lacks sample_id: {sample_id}")
        greedy = _signal(record.get("greedy", {}))
        samples = _load_sample_signals(
            generation_input,
            record,
            hidden_root=hidden_input.parent if hidden_input else None,
            hidden_descriptor=hidden_records.get(sample_id),
        )
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
                uq[name] = method.compute(
                    question=sample["question"],
                    greedy=greedy,
                    samples=[value for value in samples if value is not None],
                )
        write_sample_json_line(
            output,
            run,
            {"sample": {"sample_id": sample_id}, "uq": uq},
        )
        written += 1
    _finalize(output, methods)
    return written, skipped
