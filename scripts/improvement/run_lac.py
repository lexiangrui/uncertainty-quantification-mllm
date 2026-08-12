#!/usr/bin/env python3
"""Compute Layer-wise Answer Consistency (LAC) scores for a generation file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import iter_dataset
from src.utils import completed_sample_ids, load_jsonl_records, write_sample_json_line
from src.improvement import LacBackend


def _load_generation(path: Path) -> tuple[dict, list[dict]]:
    rows = load_jsonl_records(path)
    if not rows or rows[0].get("record_type") != "run":
        raise ValueError(f"generation input lacks a run header: {path}")
    return rows[0]["run"], rows[1:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute LAC uncertainty scores.")
    parser.add_argument("--greedy-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--family", required=True, choices=("llava_1_5", "qwen2_5_vl", "internvl3_5"))
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--dataset-source", required=True, type=Path)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-ids-file", type=Path, default=None)
    args = parser.parse_args()

    sample_ids_filter = None
    if args.sample_ids_file is not None:
        sample_ids_filter = set()
        with args.sample_ids_file.open(encoding="utf-8") as f:
            for line in f:
                sid = line.strip()
                if sid:
                    sample_ids_filter.add(sid)
        print(f"Filtering to {len(sample_ids_filter)} sample IDs")

    generation_run, records = _load_generation(args.greedy_input)
    dataset = generation_run["dataset"]

    run = {
        "lac_output_version": "layer-wise-answer-consistency-v1",
        "greedy_input": str(args.greedy_input.resolve()),
        "greedy_run": generation_run,
        "method_config": {
            "name": "lac",
            "method_version": "logit-lens-v1",
            "score_direction": "higher_is_more_uncertain",
            "description": "Layer-wise answer NLL discrepancy via Logit Lens.",
        },
    }
    completed = completed_sample_ids(args.output, run)

    backend = LacBackend(
        args.family,
        args.model_path,
        adapter_path=args.adapter_path,
        attn_implementation=args.attn_implementation,
    )
    backend._load()

    written = 0
    skipped = 0

    for sample in iter_dataset(dataset, args.dataset_source, args.limit):
        sample_id = sample.sample_id
        if sample_id in completed:
            continue
        if sample_ids_filter is not None and sample_id not in sample_ids_filter:
            continue

        record = None
        for r in records:
            if r.get("sample", {}).get("sample_id") == sample_id:
                record = r
                break
        if record is None:
            skipped += 1
            continue

        greedy = record.get("greedy", {})
        raw_response = greedy.get("raw_response", "")
        if not greedy.get("sections_valid") or not raw_response:
            skipped += 1
            continue
        if not sample.image:
            skipped += 1
            continue

        try:
            full_inputs, prompt_length, answer_span = backend.prepare_inputs(
                sample.image, sample.question, raw_response,
            )
            if full_inputs is None or answer_span is None:
                skipped += 1
                continue

            result = backend.compute_lac(full_inputs, prompt_length, answer_span)
        except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError):
            skipped += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        write_sample_json_line(args.output, run, {
            "sample": {"sample_id": sample_id},
            "uq": {"lac": result.to_dict()},
        })
        written += 1

        del full_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if written % 10 == 0:
            print(f"lac_progress written={written} skipped={skipped}", flush=True)

    print(f"completed LAC: written={written} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
