#!/usr/bin/env python3
"""Replay vLLM token artifacts with HF to materialize exact UQ signals."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import iter_dataset
from src.generation.prompt import build_prompt
from src.generation.runner import _hidden_sidecar_path, _response_record
from src.models import GenerationRequest, load_backend
from src.utils import load_jsonl_records, write_sample_json_line


def _token_payload(path: Path, record: dict) -> dict[str, torch.Tensor]:
    descriptor = record.get("generation_tokens") or {}
    relative = descriptor.get("path")
    if not isinstance(relative, str):
        raise ValueError(f"record has no token sidecar: {record.get('sample', {}).get('sample_id')}")
    sidecar = (path.parent / relative).resolve()
    if not sidecar.is_relative_to(path.parent.resolve()) or not sidecar.is_file():
        raise ValueError(f"token sidecar is missing or escapes input directory: {sidecar}")
    payload = torch.load(sidecar, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid token sidecar: {sidecar}")
    return payload


def _copy_token_sidecar(input_path: Path, output_path: Path, record: dict) -> None:
    descriptor = record.get("generation_tokens") or {}
    relative = descriptor.get("path")
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


def _request(sample, *, role: str, draw_index: int | None, seed: int, family: str) -> GenerationRequest:
    return GenerationRequest(
        request_id=f"replay:{sample.sample_id}:{role}:{draw_index if draw_index is not None else 'greedy'}",
        sample_id=sample.sample_id,
        role=role,  # type: ignore[arg-type]
        draw_index=draw_index,
        seed=seed,
        image=sample.image,
        prompt=build_prompt(sample.question, sample.image is not None, style="xml_lora"),
    )


def replay_file(
    *,
    input_path: Path,
    output_path: Path,
    dataset_source: Path,
    family: str,
    model_path: Path,
    adapter_path: Path | None,
    limit: int | None,
) -> tuple[int, int]:
    rows = load_jsonl_records(input_path)
    if not rows or rows[0].get("record_type") != "run":
        raise ValueError(f"input lacks generation run header: {input_path}")
    input_run = dict(rows[0]["run"])
    records = rows[1:]
    samples = {
        sample.sample_id: sample
        for sample in iter_dataset(input_run["dataset"], dataset_source, None)
    }
    backend = load_backend(
        family,
        model_path,
        adapter_path=adapter_path,
        engine="huggingface",
        attn_implementation=("sdpa" if family == "internvl3_5_original" else "flash_attention_2"),
    )
    run = {
        **input_run,
        "internal_state_engine": "transformers",
        "replay_model_path": str(model_path.resolve()),
        "replay_complete": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for record in records:
        sample_id = record.get("sample", {}).get("sample_id")
        sample = samples.get(sample_id)
        if sample is None:
            skipped += 1
            continue
        payload = _token_payload(input_path, record)
        updated = {**record}
        _copy_token_sidecar(input_path, output_path, record)
        if "greedy" in record:
            greedy = dict(record["greedy"])
            token_tensor = payload.get("greedy")
            if not isinstance(token_tensor, torch.Tensor):
                raise ValueError(f"missing greedy token IDs for {sample_id}")
            response = backend.teacher_force_response(
                _request(sample, role="greedy", draw_index=None, seed=int(greedy.get("generation_seed") or 0), family=family),
                tuple(int(value) for value in token_tensor.tolist()),
            )
            replay_record, _ = _response_record(backend, response, "xml")
            for key in ("signals", "sections_valid", "section_error", "vision", "reasoning", "answer"):
                if key in replay_record:
                    greedy[key] = replay_record[key]
            updated["greedy"] = greedy
        if "samples" in record:
            sample_records = []
            hidden_values = []
            for index, item in enumerate(record["samples"]):
                token_tensor = payload.get(f"sample_{index}")
                if not isinstance(token_tensor, torch.Tensor):
                    raise ValueError(f"missing sample_{index} token IDs for {sample_id}")
                response = backend.teacher_force_response(
                    _request(sample, role="sample", draw_index=index, seed=int(item.get("seed") or 0), family=family),
                    tuple(int(value) for value in token_tensor.tolist()),
                )
                replay_record, signals = _response_record(backend, response, "xml")
                current = {**item}
                for key in ("signals", "sections_valid", "section_error", "vision", "reasoning", "answer"):
                    if key in replay_record:
                        current[key] = replay_record[key]
                current["hidden_state_index"] = index if signals and signals.final_hidden else None
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
        if limit is not None and written >= limit:
            break
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay vLLM tokens with HF and materialize hidden/log-prob signals")
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
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    written, skipped = replay_file(
        input_path=args.input,
        output_path=args.output,
        dataset_source=args.dataset_source,
        family=args.model_family,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        limit=args.limit,
    )
    print(f"completed HF replay: written={written} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
