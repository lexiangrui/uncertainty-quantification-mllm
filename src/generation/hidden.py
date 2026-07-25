from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

from src.datasets import iter_dataset
from src.models.huggingface import HuggingFaceMultimodalBackend
from src.utils import completed_sample_ids, write_sample_json_line

from .parser import answer_character_span
from .prompt import build_prompt


def _load_generation(path: Path) -> tuple[dict, dict[str, dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or rows[0].get("record_type") != "run":
        raise ValueError(f"generation input lacks a run header: {path}")
    records: dict[str, dict] = {}
    for record in rows[1:]:
        sample_id = record.get("sample", {}).get("sample_id")
        if record.get("record_type") != "sample" or not isinstance(sample_id, str):
            raise ValueError(f"invalid generation record in {path}")
        if sample_id in records:
            raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
        records[sample_id] = record
    return rows[0]["run"], records


def _safe_sidecar(root: Path, descriptor: dict) -> Path:
    relative = Path(descriptor["path"])
    if relative.is_absolute():
        raise ValueError("token sidecar path must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("token sidecar path escapes generation directory")
    return path


def _token_end_for_answer(backend, text: str, token_ids: tuple[int, ...]) -> int:
    try:
        _, character_end = answer_character_span(text, "xml")
    except ValueError:
        return len(token_ids)
    for end in range(1, len(token_ids) + 1):
        decoded = backend.decode_generated_tokens(token_ids[:end])
        if len(decoded) >= character_end:
            return end
    return len(token_ids)


def _collate_inputs(backend, sample, token_sequences: list[tuple[int, ...]]):
    prompt = build_prompt(sample.question, sample.image is not None)
    prepared = [backend._inputs(sample.image, prompt) for _ in token_sequences]
    prompt_lengths = [int(value["input_ids"].shape[1]) for value in prepared]
    full_ids = [
        torch.cat(
            [
                value["input_ids"][0],
                torch.tensor(tokens, dtype=torch.long, device=backend.device),
            ]
        )
        for value, tokens in zip(prepared, token_sequences, strict=True)
    ]
    max_length = max(value.numel() for value in full_ids)
    tokenizer = getattr(backend.processor, "tokenizer", backend.processor)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    input_ids = torch.full(
        (len(full_ids), max_length),
        int(pad_id),
        dtype=torch.long,
        device=backend.device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, ids in enumerate(full_ids):
        input_ids[index, : ids.numel()] = ids
        attention_mask[index, : ids.numel()] = 1
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}
    for key in prepared[0]:
        if key in {"input_ids", "attention_mask"}:
            continue
        values = [item[key] for item in prepared]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError(f"unsupported multimodal input type for {key}")
        # The supported LLaVA/Qwen/InternVL processors represent image data as
        # a concatenation of per-request rows or patches.
        batch[key] = torch.cat(values, dim=0)
    return batch, prompt_lengths


def _extract_sample_hidden(
    backend: HuggingFaceMultimodalBackend,
    sample,
    record: dict,
    token_payload: dict,
) -> torch.Tensor:
    samples = record.get("samples", [])
    token_sequences = [
        tuple(int(value) for value in token_payload[f"sample_{index}"].tolist())
        for index in range(len(samples))
    ]
    batch, prompt_lengths = _collate_inputs(backend, sample, token_sequences)
    response_positions = []
    for index, (sample_record, token_ids) in enumerate(
        zip(samples, token_sequences, strict=True)
    ):
        token_end = _token_end_for_answer(
            backend, sample_record.get("raw_response", ""), token_ids
        )
        if token_end < 1:
            raise ValueError(f"empty generated response for {sample.sample_id}:{index}")
        response_positions.append(prompt_lengths[index] + token_end - 1)

    captured = None

    def capture(_module, _args, output):
        nonlocal captured
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, tuple) and output:
            hidden = output[0]
        if hidden is None or hidden.ndim != 3:
            raise RuntimeError("language model did not expose last_hidden_state")
        captured = hidden

    assert backend.model is not None
    handle = backend._semantic_embedding_module().register_forward_hook(capture)
    try:
        with torch.inference_mode():
            backend.model(**batch, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    if captured is None:
        raise RuntimeError("hidden-state forward pass did not reach the decoder")
    vectors = torch.stack(
        [captured[index, position] for index, position in enumerate(response_positions)]
    )
    return vectors.detach().to(device="cpu", dtype=torch.float16)


def _write_hidden(output: Path, sample_id: str, tensor: torch.Tensor) -> dict:
    directory = output.with_suffix(".hidden")
    directory.mkdir(parents=True, exist_ok=True)
    filename = hashlib.sha256(sample_id.encode()).hexdigest()[:16] + ".pt"
    path = directory / filename
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    try:
        torch.save(tensor, temporary)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return {
        "path": str(path.relative_to(output.parent)),
        "shape": list(tensor.shape),
        "dtype": "float16",
        "role": "samples",
        "position": "answer_last_token",
    }


def extract_hidden_states(
    *,
    generation_input: Path,
    output: Path,
    family: str,
    model_path: Path,
    adapter_path: Path,
    dataset_source: Path,
    attn_implementation: str | None,
) -> tuple[int, int]:
    generation_run, records = _load_generation(generation_input)
    dataset = generation_run["dataset"]
    run = {
        "hidden_output_version": "offline-answer-hidden-v1",
        "generation_input": str(generation_input.resolve()),
        "generation_run": generation_run,
        "model_family": family,
        "model_path": str(model_path.resolve()),
        "adapter_path": str(adapter_path.resolve()),
        "position": "answer_last_token",
    }
    completed = completed_sample_ids(output, run)
    pending = set(records) - completed
    backend = HuggingFaceMultimodalBackend(
        family,
        model_path,
        attn_implementation=attn_implementation,
        adapter_path=adapter_path,
    )
    backend._load()
    written = 0
    for sample in iter_dataset(dataset, dataset_source, None):
        if sample.sample_id not in pending:
            continue
        record = records[sample.sample_id]
        token_path = _safe_sidecar(
            generation_input.parent, record["generation_tokens"]
        )
        payload = torch.load(token_path, map_location="cpu", weights_only=True)
        tensor = _extract_sample_hidden(backend, sample, record, payload)
        descriptor = _write_hidden(output, sample.sample_id, tensor)
        write_sample_json_line(
            output,
            run,
            {
                "sample": {"sample_id": sample.sample_id},
                "hidden_states": descriptor,
            },
        )
        pending.remove(sample.sample_id)
        written += 1
        if written % 10 == 0:
            print(f"hidden_progress written={written} remaining={len(pending)}", flush=True)
        if not pending:
            break
    if pending:
        raise ValueError(f"dataset lacks {len(pending)} generated sample ids")
    return written, len(completed)
