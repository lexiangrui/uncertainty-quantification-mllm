#!/usr/bin/env python3
"""Batch-replay exact vLLM tokens with HF to materialize UQ artifacts."""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import iter_dataset
from src.generation.prompt import build_prompt
from src.generation.runner import _hidden_sidecar_path, _response_record
from src.models import GenerationRequest, load_replay_backend
from src.models.huggingface import HuggingFaceReplayBackend
from src.models.runtime import replay_batch_size, visible_gpu_memory_gib
from src.utils import completed_sample_ids, load_jsonl_records, write_sample_json_line


def _replay_attention_implementation(family: str) -> str:
    """Select the attention implementation supported by each model package."""

    if family != "internvl3_5_original":
        return "flash_attention_2"
    try:
        from transformers.utils import is_flash_attn_2_available

        if is_flash_attn_2_available():
            return "flash_attention_2"
    except Exception:
        pass
    # InternVL's original remote code maps use_flash_attn=False to eager and
    # overwrites the nested language-model config during construction.
    return "eager"


def _chunks(values: list[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _load_sample_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value]
    if not values:
        raise ValueError(f"sample ID file is empty: {path}")
    sample_ids = set(values)
    if len(sample_ids) != len(values):
        raise ValueError(f"sample ID file contains duplicates: {path}")
    return sample_ids


def _token_payload(path: Path, record: dict) -> dict[str, torch.Tensor]:
    descriptor = record.get("generation_tokens") or {}
    relative = descriptor.get("path")
    if not isinstance(relative, str):
        raise ValueError(
            f"record has no token sidecar: {record.get('sample', {}).get('sample_id')}"
        )
    sidecar = (path.parent / relative).resolve()
    if not sidecar.is_relative_to(path.parent.resolve()) or not sidecar.is_file():
        raise ValueError(f"token sidecar is missing or escapes input directory: {sidecar}")
    payload = torch.load(sidecar, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid token sidecar: {sidecar}")
    return payload


def _copy_token_sidecar(input_path: Path, output_path: Path, record: dict) -> None:
    relative = (record.get("generation_tokens") or {}).get("path")
    if not isinstance(relative, str):
        raise ValueError("generation record has no token sidecar descriptor")
    source = (input_path.parent / relative).resolve()
    target = (output_path.parent / relative).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not target.is_relative_to(output_path.parent.resolve()):
        raise ValueError(f"token sidecar target escapes output directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source != target:
        shutil.copy2(source, target)


def _request(sample, *, role: str, draw_index: int | None, seed: int) -> GenerationRequest:
    return GenerationRequest(
        request_id=f"replay:{sample.sample_id}:{role}:{draw_index if draw_index is not None else 'greedy'}",
        sample_id=sample.sample_id,
        role=role,  # type: ignore[arg-type]
        draw_index=draw_index,
        seed=seed,
        image=sample.image,
        prompt=build_prompt(sample.question, sample.image is not None, style="xml_lora"),
    )


def _tokens(payload: dict[str, torch.Tensor], key: str, sample_id: str) -> tuple[int, ...]:
    tensor = payload.get(key)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"missing {key} token IDs for {sample_id}")
    return tuple(int(value) for value in tensor.tolist())


def _canonical_response(response, generation_record: dict):
    """Keep exact replay tensors while parsing vLLM's stop-trimmed output text."""
    canonical_text = generation_record.get("raw_response")
    if not isinstance(canonical_text, str) or not canonical_text:
        raise ValueError("generation record has no canonical raw_response")
    return replace(response, text=canonical_text)


def _run_replay_calls(
    backend: HuggingFaceReplayBackend,
    calls: list[tuple[GenerationRequest, tuple[int, ...]]],
    batch_size: int,
) -> dict:
    """Replay homogeneous modality batches; HallusionBench changes modality mid-file."""
    responses = {}
    for has_image in (True, False):
        modality_calls = [
            item for item in calls if (item[0].image is not None) == has_image
        ]
        for call_batch in _chunks(modality_calls, batch_size):
            requests = [item[0] for item in call_batch]
            sequences = [item[1] for item in call_batch]
            responses.update(backend.teacher_force_responses(requests, sequences))
    return responses


def replay_file(
    *,
    input_path: Path,
    output_path: Path,
    dataset_source: Path,
    family: str,
    model_path: Path,
    adapter_path: Path | None,
    batch_size: int,
    limit: int | None,
    sample_ids_filter: set[str] | None = None,
    backend: HuggingFaceReplayBackend | None = None,
) -> tuple[int, int]:
    rows = load_jsonl_records(input_path)
    if not rows or rows[0].get("record_type") != "run":
        raise ValueError(f"input lacks generation run header: {input_path}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    input_run = dict(rows[0]["run"])
    records = rows[1:]
    if sample_ids_filter is not None:
        records = [
            record
            for record in records
            if record.get("sample", {}).get("sample_id") in sample_ids_filter
        ]
    if limit is not None:
        records = records[:limit]
    samples = {
        sample.sample_id: sample
        for sample in iter_dataset(input_run["dataset"], dataset_source, None)
    }
    if backend is None:
        backend = load_replay_backend(
            family,
            model_path,
            adapter_path=adapter_path,
            attn_implementation=_replay_attention_implementation(family),
        )
    run = {
        **input_run,
        "internal_state_engine": "hf_teacher_forcing",
        "replay_model_path": str(model_path.resolve()),
        "replay_batch_size": batch_size,
        "replay_complete": True,
    }
    if family == "internvl3_5_original":
        run["replay_image_preprocessing"] = "vllm_internvl_dynamic_tiles_v1"
    if sample_ids_filter is not None:
        run["sample_filter"] = sorted(sample_ids_filter)
    completed = completed_sample_ids(output_path, run)
    pending = [
        record
        for record in records
        if record.get("sample", {}).get("sample_id") not in completed
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = len(completed)

    for record_batch in _chunks(pending, batch_size):
        prepared: list[tuple[dict, Any, dict[str, torch.Tensor]]] = []
        calls: list[tuple[GenerationRequest, tuple[int, ...]]] = []
        for record in record_batch:
            sample_id = record.get("sample", {}).get("sample_id")
            sample = samples.get(sample_id)
            if sample is None:
                skipped += 1
                continue
            payload = _token_payload(input_path, record)
            prepared.append((record, sample, payload))
            if "greedy" in record:
                greedy = record["greedy"]
                calls.append(
                    (
                        _request(
                            sample,
                            role="greedy",
                            draw_index=None,
                            seed=int(greedy.get("generation_seed") or 0),
                        ),
                        _tokens(payload, "greedy", sample_id),
                    )
                )
            for index, item in enumerate(record.get("samples", [])):
                calls.append(
                    (
                        _request(
                            sample,
                            role="sample",
                            draw_index=index,
                            seed=int(item.get("seed") or 0),
                        ),
                        _tokens(payload, f"sample_{index}", sample_id),
                    )
                )

        responses = _run_replay_calls(backend, calls, batch_size)

        for record, sample, _payload in prepared:
            sample_id = sample.sample_id
            updated = {**record}
            _copy_token_sidecar(input_path, output_path, record)
            if "greedy" in record:
                greedy = dict(record["greedy"])
                response = responses[f"replay:{sample_id}:greedy:greedy"]
                response = _canonical_response(response, greedy)
                replay_record, _ = _response_record(backend, response, "xml")
                for key in (
                    "signals", "sections_valid", "section_error",
                    "vision", "reasoning", "answer",
                ):
                    if key in replay_record:
                        greedy[key] = replay_record[key]
                updated["greedy"] = greedy
            if "samples" in record:
                sample_records = []
                hidden_values = []
                for index, item in enumerate(record["samples"]):
                    response = responses[f"replay:{sample_id}:sample:{index}"]
                    response = _canonical_response(response, item)
                    replay_record, signals = _response_record(backend, response, "xml")
                    current = {**item}
                    for key in (
                        "signals", "sections_valid", "section_error",
                        "vision", "reasoning", "answer",
                    ):
                        if key in replay_record:
                            current[key] = replay_record[key]
                    current["hidden_state_index"] = (
                        index if signals and signals.final_hidden else None
                    )
                    if signals and signals.final_hidden:
                        hidden_values.append(signals.final_hidden)
                    sample_records.append(current)
                updated["samples"] = sample_records
                if len(hidden_values) == len(sample_records) and hidden_values:
                    tensor = torch.tensor(hidden_values, dtype=torch.float16)
                    hidden_path, descriptor = _hidden_sidecar_path(output_path, sample_id)
                    hidden_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(tensor, hidden_path)
                    updated["hidden_states"] = {
                        **descriptor,
                        "shape": list(tensor.shape),
                        "dtype": "float16",
                        "role": "samples",
                        "position": "answer_last_token",
                        "source": "hf_teacher_forcing_replay",
                    }
            write_sample_json_line(output_path, run, updated)
            written += 1
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-replay vLLM tokens with HF and materialize hidden/log-prob signals"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-source", required=True, type=Path)
    parser.add_argument(
        "--model-family",
        required=True,
        choices=("llava_1_5", "qwen2_5_vl", "internvl3_5", "internvl3_5_original"),
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument(
        "--batch-size", type=int, default=0,
        help="HF replay batch; 0 selects from visible GPU memory",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-ids-file", type=Path, default=None)
    args = parser.parse_args()
    sample_ids_filter = _load_sample_ids(args.sample_ids_file)
    memory_gib = visible_gpu_memory_gib()
    batch_size = args.batch_size or replay_batch_size(memory_gib)
    print(f"HF replay GPU memory={memory_gib:.1f} GiB batch_size={batch_size}")
    written, skipped = replay_file(
        input_path=args.input,
        output_path=args.output,
        dataset_source=args.dataset_source,
        family=args.model_family,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        batch_size=batch_size,
        limit=args.limit,
        sample_ids_filter=sample_ids_filter,
    )
    print(f"completed HF replay: written={written} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
