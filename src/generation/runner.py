from __future__ import annotations

import hashlib
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from src.datasets import BenchmarkSample, iter_dataset
from src.models import (
    GeneratedResponse,
    GenerationBackend,
    GenerationRequest,
)
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
    final_hidden: tuple[float, ...] = ()


@dataclass
class DrawState:
    attempts_used: int = 0
    attempt_seeds: list[int] = field(default_factory=list)
    record: dict | None = None
    signals: ResponseSignals | None = None
    response: GeneratedResponse | None = None
    done: bool = False


@dataclass
class SampleState:
    sample: BenchmarkSample
    greedy_record: dict | None = None
    greedy_signals: ResponseSignals | None = None
    greedy_response: GeneratedResponse | None = None
    draws: list[DrawState] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.greedy_record is not None and all(draw.done for draw in self.draws)


def _serialized_signals(signals: ResponseSignals | None) -> dict | None:
    if signals is None:
        return None
    return {
        "token_count": signals.token_count,
        "sequence_log_prob": signals.sequence_log_prob,
        "mean_log_prob": signals.mean_log_prob,
        "sampling_sequence_log_prob": signals.sampling_sequence_log_prob,
    }


def _derived_seed(seed: int, sample_id: str, draw: int, attempt: int = 0) -> int:
    value = f"{seed}:{sample_id}:{draw}:{attempt}"
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
        decoded_lengths.append(
            len(backend.decode_generated_tokens(response.token_ids[:end]))
        )
    token_start = next(
        (
            index - 1
            for index, length in enumerate(decoded_lengths[1:], start=1)
            if length > character_start
        ),
        len(response.token_ids),
    )
    token_end = next(
        (
            index
            for index, length in enumerate(decoded_lengths[1:], start=1)
            if length >= character_end
        ),
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
    base = {
        "raw_response": response.text,
        "finish_reason": response.finish_reason,
    }
    try:
        parsed = parse_structured_response(response.text, response_format)
    except ValueError as error:
        return (
            {
                **base,
                "sections_valid": False,
                "section_error": str(error),
                "vision": None,
                "reasoning": None,
                "answer": None,
                "signals": None,
            },
            None,
        )
    character_start, character_end = answer_character_span(
        response.text, response_format
    )
    token_start, token_end = _token_span_for_characters(
        backend, response, character_start, character_end
    )
    answer_log_probs = response.token_log_probs[token_start:token_end]
    sampling_log_probs = response.sampling_token_log_probs[token_start:token_end]
    if not answer_log_probs or not sampling_log_probs:
        raise ValueError("answer has no aligned token log probabilities")
    sequence_log_prob = float(sum(answer_log_probs))
    mean_log_prob = sequence_log_prob / len(answer_log_probs)
    sampling_sequence_log_prob = float(sum(sampling_log_probs))
    if not all(
        math.isfinite(value)
        for value in (
            sequence_log_prob,
            mean_log_prob,
            sampling_sequence_log_prob,
        )
    ):
        raise ValueError("answer log probability is not finite")
    signals = ResponseSignals(
        answer=parsed.answer,
        token_count=len(answer_log_probs),
        sequence_log_prob=sequence_log_prob,
        mean_log_prob=mean_log_prob,
        sampling_sequence_log_prob=sampling_sequence_log_prob,
    )
    return (
        {
            **base,
            "sections_valid": True,
            "section_error": None,
            **parsed.to_dict(),
            "signals": _serialized_signals(signals),
        },
        signals,
    )


def _token_directory(output: Path) -> Path:
    return output.with_suffix(".tokens")


def _sidecar_filename(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode()).hexdigest()[:16] + ".pt"


def _write_token_sidecar(output: Path, state: SampleState) -> dict:
    if state.greedy_response is None or any(draw.response is None for draw in state.draws):
        raise RuntimeError(f"incomplete token state for {state.sample.sample_id}")
    directory = _token_directory(output)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _sidecar_filename(state.sample.sample_id)
    payload = {
        "greedy": torch.tensor(state.greedy_response.token_ids, dtype=torch.int32),
        **{
            f"sample_{index}": torch.tensor(draw.response.token_ids, dtype=torch.int32)
            for index, draw in enumerate(state.draws)
            if draw.response is not None
        },
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return {
        "path": str(path.relative_to(output.parent)),
        "format": "generated-token-ids-v1",
        "keys": ["greedy"] + [f"sample_{index}" for index in range(len(state.draws))],
    }


def _new_state(sample: BenchmarkSample, num_samples: int) -> SampleState:
    return SampleState(sample=sample, draws=[DrawState() for _ in range(num_samples)])


def _request(
    state: SampleState,
    *,
    role: str,
    draw_index: int | None,
    attempt: int,
    seed: int,
    prompt_style: str,
) -> GenerationRequest:
    draw_label = "greedy" if draw_index is None else f"sample-{draw_index}"
    return GenerationRequest(
        request_id=f"{state.sample.sample_id}:{draw_label}:attempt-{attempt}",
        sample_id=state.sample.sample_id,
        role=role,  # type: ignore[arg-type]
        draw_index=draw_index,
        seed=seed,
        image=state.sample.image,
        prompt=build_prompt(
            state.sample.question,
            state.sample.image is not None,
            style=prompt_style,
        ),
    )


def _initial_requests(
    state: SampleState, *, seed: int, prompt_style: str
) -> list[GenerationRequest]:
    requests = [
        _request(
            state,
            role="greedy",
            draw_index=None,
            attempt=1,
            seed=_derived_seed(seed, state.sample.sample_id, -1),
            prompt_style=prompt_style,
        )
    ]
    for index in range(len(state.draws)):
        requests.append(
            _request(
                state,
                role="sample",
                draw_index=index,
                attempt=1,
                seed=_derived_seed(seed, state.sample.sample_id, index),
                prompt_style=prompt_style,
            )
        )
    return requests


def _record(state: SampleState, output: Path, reject_resample_k: int) -> dict:
    if not state.done or state.greedy_record is None:
        raise RuntimeError(f"sample is incomplete: {state.sample.sample_id}")
    samples: list[dict] = []
    accepted_samples = 0
    total_attempts = 0
    rejected_attempts = 0
    for index, draw in enumerate(state.draws):
        assert draw.record is not None
        accepted = bool(draw.record["sections_valid"])
        accepted_samples += int(accepted)
        total_attempts += draw.attempts_used
        rejected_attempts += draw.attempts_used - int(accepted)
        samples.append(
            {
                "index": index,
                "role": "sample",
                "temperature": 1.0,
                "seed": draw.attempt_seeds[-1],
                "attempt_seeds": draw.attempt_seeds,
                **draw.record,
                "reject_resample": {
                    "max_attempts": reject_resample_k,
                    "attempts_used": draw.attempts_used,
                    "rejected_count": draw.attempts_used - int(accepted),
                    "accepted": accepted,
                },
            }
        )
    return {
        "sample": _serialize_sample(state.sample),
        "greedy": state.greedy_record,
        "samples": samples,
        "generation_tokens": _write_token_sidecar(output, state),
        "reject_resample_summary": {
            "max_attempts": reject_resample_k,
            "retained_samples": len(state.draws),
            "accepted_samples": accepted_samples,
            "failed_samples": len(state.draws) - accepted_samples,
            "total_attempts": total_attempts,
            "rejected_attempts": rejected_attempts,
        },
    }


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
    prompt_style: str = "xml_lora",
    reject_resample_k: int = 10,
    max_batch_size: int = 5,
    request_window_samples: int = 16,
) -> tuple[int, int]:
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if reject_resample_k < 1:
        raise ValueError("reject_resample_k must be positive")
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")
    if request_window_samples < 1:
        raise ValueError("request_window_samples must be positive")
    prompt_spec = get_prompt_spec(prompt_style)
    run = {
        "dataset": dataset,
        "dataset_source": str(dataset_source.resolve()),
        "model_family": family,
        "model_id": backend.model_id,
        "model_path": str(model_path.resolve()),
        "model_runtime": backend.runtime_config,
        "prompt_version": prompt_spec.version,
        "greedy": {"do_sample": False, "temperature": 0.0, "retry": False},
        "sampling": {
            "do_sample": True,
            "temperature": 1.0,
            "num_samples": num_samples,
            "reject_resample_k": reject_resample_k,
        },
        "scheduler": {
            "type": "vllm_continuous_batching",
            "max_batch_size": max_batch_size,
            "request_window_samples": request_window_samples,
            "mixed_greedy_and_sampling": True,
        },
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "limit": limit,
        "generation_output_version": "responses-jsonl-token-ids-v2",
        "hidden_state_execution": "separate",
        "uq_execution": "deferred",
    }
    completed = completed_sample_ids(output, run)
    source = (
        sample
        for sample in iter_dataset(dataset, dataset_source, limit)
        if sample.sample_id not in completed
    )
    active: dict[str, SampleState] = {}
    pending: list[GenerationRequest] = []
    exhausted = False
    written = 0
    skipped = len(completed)
    generation_seconds = 0.0
    write_seconds = 0.0

    def fill_window() -> None:
        nonlocal exhausted
        while not exhausted and len(active) < request_window_samples:
            try:
                sample = next(source)
            except StopIteration:
                exhausted = True
                break
            state = _new_state(sample, num_samples)
            active[sample.sample_id] = state
            pending.extend(
                _initial_requests(state, seed=seed, prompt_style=prompt_style)
            )

    fill_window()
    while active:
        if not pending:
            raise RuntimeError("dynamic generation queue stalled with active samples")
        started = time.perf_counter()
        generated = backend.generate_requests(
            pending, max_new_tokens=max_new_tokens
        )
        generation_seconds += time.perf_counter() - started
        retries: list[GenerationRequest] = []
        for request in pending:
            response = generated.get(request.request_id)
            if response is None:
                raise RuntimeError(f"missing generated response: {request.request_id}")
            state = active[request.sample_id]
            record, signals = _response_record(
                backend, response, prompt_spec.response_format
            )
            if request.role == "greedy":
                state.greedy_record = record
                state.greedy_signals = signals
                state.greedy_response = response
                continue
            assert request.draw_index is not None
            draw = state.draws[request.draw_index]
            draw.attempts_used += 1
            draw.attempt_seeds.append(request.seed)
            accepted = bool(record["sections_valid"])
            if not accepted and draw.attempts_used < reject_resample_k:
                next_attempt = draw.attempts_used + 1
                retries.append(
                    _request(
                        state,
                        role="sample",
                        draw_index=request.draw_index,
                        attempt=next_attempt,
                        seed=_derived_seed(
                            seed,
                            state.sample.sample_id,
                            request.draw_index,
                            draw.attempts_used,
                        ),
                        prompt_style=prompt_style,
                    )
                )
                continue
            draw.record = record
            draw.signals = signals
            draw.response = response
            draw.done = True

        started = time.perf_counter()
        for sample_id, state in list(active.items()):
            if not state.done:
                continue
            write_sample_json_line(
                output, run, _record(state, output, reject_resample_k)
            )
            del active[sample_id]
            written += 1
        write_seconds += time.perf_counter() - started
        pending = retries
        fill_window()
        if written and written % 10 == 0:
            print(
                "generation_timing "
                f"written={written} generation={generation_seconds:.3f}s "
                f"write={write_seconds:.3f}s active={len(active)} "
                f"queued_requests={len(pending)}",
                flush=True,
            )
    print(
        "generation_timing_final "
        f"written={written} skipped={skipped} generation={generation_seconds:.3f}s "
        f"write={write_seconds:.3f}s",
        flush=True,
    )
    return written, skipped
