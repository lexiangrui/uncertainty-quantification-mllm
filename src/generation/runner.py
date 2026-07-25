from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import torch

from src.datasets import BenchmarkSample, iter_dataset
from src.models import GeneratedResponse, GenerationBackend
from src.utils import completed_sample_ids, write_sample_json_line

from .parser import answer_character_span, parse_structured_response
from .prompt import build_prompt, get_prompt_spec


@dataclass(frozen=True)
class ResponseSignals:
    answer: str
    token_count: int
    sequence_log_prob: float
    mean_log_prob: float
    sampling_sequence_log_prob: float
    final_hidden: tuple[float, ...]


def _serialized_signals(signals: ResponseSignals | None) -> dict | None:
    if signals is None:
        return None
    return {
        "token_count": signals.token_count,
        "sequence_log_prob": signals.sequence_log_prob,
        "mean_log_prob": signals.mean_log_prob,
        "sampling_sequence_log_prob": signals.sampling_sequence_log_prob,
    }


def _hidden_directory(output: Path) -> Path:
    return output.with_suffix(".hidden")


def _hidden_filename(sample_id: str) -> str:
    digest = hashlib.sha256(sample_id.encode()).hexdigest()[:16]
    return f"{digest}.pt"


def _write_hidden_states(
    output: Path, sample_id: str, signals: list[ResponseSignals | None]
) -> dict | None:
    if any(value is None for value in signals):
        return None
    tensor = torch.tensor(
        [value.final_hidden for value in signals if value is not None],
        dtype=torch.float16,
    )
    directory = _hidden_directory(output)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _hidden_filename(sample_id)
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
    }


class OnlineUQMethod(Protocol):
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


def _derived_seed(seed: int, sample_id: str, draw: int, attempt: int = 0) -> int:
    value = f"{seed}:{sample_id}:{draw}"
    if attempt:
        value = f"{value}:{attempt}"
    digest = hashlib.sha256(value.encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _serialize_sample(sample: BenchmarkSample) -> dict:
    value = asdict(sample)
    value.pop("image")
    value["references"] = list(sample.references)
    value["has_image"] = sample.image is not None
    return value


def _token_span_for_characters(
    backend: GenerationBackend,
    response: GeneratedResponse,
    character_start: int,
    character_end: int,
) -> tuple[int, int]:
    decoded_lengths = [0]
    for end in range(1, len(response.token_ids) + 1):
        decoded = backend.decode_generated_tokens(response.token_ids[:end])
        decoded_lengths.append(len(decoded))
    token_start = next(
        (index - 1 for index, length in enumerate(decoded_lengths[1:], start=1) if length > character_start),
        len(response.token_ids),
    )
    token_end = next(
        (index for index, length in enumerate(decoded_lengths[1:], start=1) if length >= character_end),
        len(response.token_ids),
    )
    if token_start >= token_end:
        raise ValueError("answer text does not align to generated tokens")
    return token_start, token_end


def _response_record(
    backend: GenerationBackend,
    response: GeneratedResponse,
    response_format: str,
) -> tuple[dict, ResponseSignals | None]:
    text = response.text
    try:
        parsed = parse_structured_response(text, response_format)
    except ValueError as error:
        return (
            {
                "raw_response": text,
                "sections_valid": False,
                "section_error": str(error),
                "vision": None,
                "reasoning": None,
                "answer": None,
            },
            None,
        )
    character_start, character_end = answer_character_span(text, response_format)
    token_start, token_end = _token_span_for_characters(
        backend, response, character_start, character_end
    )
    answer_log_probs = response.token_log_probs[token_start:token_end]
    sampling_answer_log_probs = response.sampling_token_log_probs[
        token_start:token_end
    ]
    sequence_log_prob = float(sum(answer_log_probs))
    mean_log_prob = float(sum(answer_log_probs) / len(answer_log_probs))
    if not math.isfinite(sequence_log_prob) or not math.isfinite(mean_log_prob):
        raise ValueError("answer log probability is not finite")
    sampling_sequence_log_prob = float(sum(sampling_answer_log_probs))
    if not math.isfinite(sampling_sequence_log_prob):
        raise ValueError("answer sampling log probability is not finite")
    return (
        {
            "raw_response": text,
            "sections_valid": True,
            "section_error": None,
            **parsed.to_dict(),
        },
        ResponseSignals(
            answer=parsed.answer,
            token_count=len(answer_log_probs),
            sequence_log_prob=sequence_log_prob,
            mean_log_prob=mean_log_prob,
            sampling_sequence_log_prob=sampling_sequence_log_prob,
            final_hidden=response.final_hidden,
        ),
    )


def _finalize_uq_output(output: Path, uq_methods: tuple[OnlineUQMethod, ...]) -> None:
    finalizers = [method for method in uq_methods if hasattr(method, "finalize")]
    if not finalizers or not output.exists():
        return
    header = None
    records = []
    with output.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("record_type") == "run":
                header = record
            else:
                records.append(record)
    for method in finalizers:
        name = method.runtime_config["name"]
        method.finalize([record["uq"][name] for record in records])
    descriptor, temporary = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            if header is not None:
                handle.write(
                    json.dumps(header, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def run_generation(
    *,
    backend: GenerationBackend,
    family: str,
    model_path: Path,
    dataset: str,
    dataset_source: Path,
    output: Path,
    max_new_tokens: int,
    num_samples: int,
    seed: int,
    limit: int | None,
    uq_methods: tuple[OnlineUQMethod, ...],
    prompt_style: str = "xml_lora",
    reject_resample_k: int = 10,
    greedy_reject_resample_k: int = 10,
    greedy_recovery_temperature: float = 0.2,
    sampling_batch_size: int = 1,
) -> tuple[int, int]:
    if reject_resample_k < 1:
        raise ValueError("reject_resample_k must be positive")
    if greedy_reject_resample_k < 1:
        raise ValueError("greedy_reject_resample_k must be positive")
    if greedy_recovery_temperature <= 0:
        raise ValueError("greedy_recovery_temperature must be positive")
    if sampling_batch_size < 1:
        raise ValueError("sampling_batch_size must be positive")
    prompt_spec = get_prompt_spec(prompt_style)
    run = {
        "dataset": dataset,
        "dataset_source": str(dataset_source.resolve()),
        "model_family": family,
        "model_id": backend.model_id,
        "model_path": str(model_path.resolve()),
        "model_runtime": backend.runtime_config,
        "prompt_version": prompt_spec.version,
        "greedy": {
            "do_sample": False,
            "temperature": 0,
            "reject_resample_k": greedy_reject_resample_k,
            "recovery_strategy": "low_temperature_sampling",
            "recovery_temperature": greedy_recovery_temperature,
        },
        "sampling": {
            "do_sample": True,
            "temperature": 1.0,
            "top_p": None,
            "top_k": None,
            "num_samples": num_samples,
            "reject_resample_k": reject_resample_k,
            "batch_size": sampling_batch_size,
        },
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "limit": limit,
        "generation_output_version": "responses-jsonl-hidden-pt-v1",
        "uq_execution": (
            "online-compatibility" if uq_methods else "deferred"
        ),
        "uq_methods": [method.runtime_config for method in uq_methods],
    }
    completed = completed_sample_ids(output, run)
    written = 0
    skipped = 0
    timing = {"greedy": 0.0, "sampling": 0.0, "uq": 0.0, "write": 0.0}
    for sample in iter_dataset(dataset, dataset_source, limit):
        if sample.sample_id in completed:
            skipped += 1
            continue
        prompt = build_prompt(
            sample.question, sample.image is not None, style=prompt_style
        )
        stage_started = time.perf_counter()
        greedy = None
        greedy_signals = None
        greedy_seed = None
        greedy_attempts_used = 0
        for attempt in range(greedy_reject_resample_k):
            recovery = attempt > 0
            if recovery:
                greedy_seed = _derived_seed(seed, sample.sample_id, -1, attempt)
                torch.manual_seed(greedy_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(greedy_seed)
            greedy_response = backend.generate(
                sample.image,
                prompt,
                do_sample=recovery,
                temperature=greedy_recovery_temperature if recovery else None,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
            )[0]
            greedy, greedy_signals = _response_record(
                backend, greedy_response, prompt_spec.response_format
            )
            greedy_attempts_used = attempt + 1
            if greedy["sections_valid"]:
                break
        greedy_accepted = bool(greedy["sections_valid"])
        greedy["reject_resample"] = {
            "max_attempts": greedy_reject_resample_k,
            "attempts_used": greedy_attempts_used,
            "rejected_count": greedy_attempts_used - int(greedy_accepted),
            "accepted": greedy_accepted,
            "initial_strategy": "greedy",
            "accepted_strategy": (
                "greedy"
                if greedy_accepted and greedy_attempts_used == 1
                else "low_temperature_sampling"
                if greedy_accepted
                else None
            ),
        }
        if greedy_seed is not None:
            greedy["recovery_seed"] = greedy_seed
        greedy["signals"] = _serialized_signals(greedy_signals)
        timing["greedy"] += time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        sampled_records = []
        sampled_signals = []
        accepted_samples = 0
        total_attempts = 0
        rejected_attempts = 0
        pending = list(range(num_samples))
        attempts = [0] * num_samples
        while pending:
            batch_indexes = pending[:sampling_batch_size]
            pending = pending[sampling_batch_size:]
            batch_seed = _derived_seed(
                seed,
                sample.sample_id,
                batch_indexes[0],
                attempts[batch_indexes[0]],
            )
            torch.manual_seed(batch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(batch_seed)
            responses = backend.generate(
                    sample.image,
                    prompt,
                    do_sample=True,
                    temperature=1.0,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=len(batch_indexes),
                )
            for index, response in zip(batch_indexes, responses, strict=True):
                attempts[index] += 1
                response_record, signals = _response_record(
                    backend, response, prompt_spec.response_format
                )
                accepted = bool(response_record["sections_valid"])
                if not accepted:
                    rejected_attempts += 1
                    if attempts[index] < reject_resample_k:
                        pending.append(index)
                        continue
                accepted_samples += int(accepted)
                total_attempts += attempts[index]
                sampled_records.append({
                    "index": index,
                    "seed": batch_seed,
                    **response_record,
                    "reject_resample": {
                        "max_attempts": reject_resample_k,
                        "attempts_used": attempts[index],
                        "rejected_count": attempts[index] - int(accepted),
                        "accepted": accepted,
                    },
                })
                sampled_signals.append((index, signals))
        sampled_records.sort(key=lambda value: value["index"])
        sampled_signals.sort(key=lambda value: value[0])
        ordered_sampled_signals = [value for _, value in sampled_signals]
        for response_record, signals in zip(
            sampled_records, ordered_sampled_signals, strict=True
        ):
            response_record["signals"] = _serialized_signals(signals)
        hidden_states = _write_hidden_states(
            output, sample.sample_id, ordered_sampled_signals
        )
        if hidden_states is not None:
            for index, response_record in enumerate(sampled_records):
                response_record["hidden_state_index"] = index
        timing["sampling"] += time.perf_counter() - stage_started

        stage_started = time.perf_counter()
        uq = {}
        for method in uq_methods:
            missing = None
            if method.required_responses in {"greedy", "greedy_and_samples"}:
                if greedy_signals is None:
                    missing = "greedy response cannot be separated into three parts"
            if method.required_responses in {"samples", "greedy_and_samples"}:
                if any(value is None for value in ordered_sampled_signals):
                    missing = "one or more sampled responses cannot be separated into three parts"
            if missing is not None:
                uq[method.runtime_config["name"]] = {
                    "valid": False,
                    "error": missing,
                    "score": None,
                }
            else:
                uq[method.runtime_config["name"]] = method.compute(
                    question=sample.question,
                    greedy=greedy_signals,
                    samples=[value for value in ordered_sampled_signals if value is not None],
                )
        timing["uq"] += time.perf_counter() - stage_started

        record = {
            "sample": _serialize_sample(sample),
            "greedy": greedy,
            "samples": sampled_records,
            "hidden_states": hidden_states,
            "reject_resample_summary": {
                "max_attempts": reject_resample_k,
                "retained_samples": num_samples,
                "accepted_samples": accepted_samples,
                "failed_samples": num_samples - accepted_samples,
                "total_attempts": total_attempts,
                "rejected_attempts": rejected_attempts,
            },
        }
        if uq_methods:
            record["uq"] = uq
        stage_started = time.perf_counter()
        write_sample_json_line(output, run, record)
        timing["write"] += time.perf_counter() - stage_started
        written += 1
        if written % 10 == 0:
            elapsed = sum(timing.values())
            print(
                "generation_timing "
                f"written={written} elapsed={elapsed:.3f}s "
                + " ".join(f"{name}={value:.3f}s" for name, value in timing.items()),
                flush=True,
            )
    _finalize_uq_output(output, uq_methods)
    elapsed = sum(timing.values())
    print(
        "generation_timing_final "
        f"written={written} skipped={skipped} elapsed={elapsed:.3f}s "
        + " ".join(f"{name}={value:.3f}s" for name, value in timing.items()),
        flush=True,
    )
    return written, skipped
