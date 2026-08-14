#!/usr/bin/env python3
"""Compute Visual Grounding Score (VGS) per sample via a single forward pass."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import iter_dataset
from src.utils import completed_sample_ids, load_jsonl_records, write_sample_json_line
from src.improvement import VgsBackend
from src.improvement.vgs import compute_vgs


def _load_generation(path):
    rows = load_jsonl_records(path)
    return rows[0]["run"], rows[1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--family", required=True, choices=("llava_1_5", "qwen2_5_vl", "internvl3_5"))
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--dataset-source", required=True, type=Path)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-ids-file", type=Path, default=None)
    args = parser.parse_args()

    sample_ids_filter = None
    if args.sample_ids_file:
        sample_ids_filter = {l.strip() for l in args.sample_ids_file.open() if l.strip()}
        print(f"Filtering to {len(sample_ids_filter)} IDs")

    gen_run, records = _load_generation(args.greedy_input)
    dataset = gen_run["dataset"]
    run = {"vgs_output_version": "v1", "greedy_input": str(args.greedy_input.resolve()), "greedy_run": gen_run}
    completed = completed_sample_ids(args.output, run)

    backend = VgsBackend(args.family, args.model_path,
                          adapter_path=args.adapter_path,
                          attn_implementation=args.attn_implementation)
    backend._load()

    written = skipped = 0
    for sample in iter_dataset(dataset, args.dataset_source, args.limit):
        sid = sample.sample_id
        if sid in completed or (sample_ids_filter and sid not in sample_ids_filter):
            continue
        record = next((r for r in records if r.get("sample", {}).get("sample_id") == sid), None)
        if not record:
            print(f"skip {sid}: generation record missing", flush=True)
            skipped += 1
            continue
        greedy = record.get("greedy", {})
        if not greedy.get("sections_valid") or not greedy.get("raw_response") or not sample.image:
            print(
                f"skip {sid}: sections_valid={greedy.get('sections_valid')} "
                f"raw_response={bool(greedy.get('raw_response'))} image={bool(sample.image)}",
                flush=True,
            )
            skipped += 1
            continue
        try:
            full_inputs, prompt_length, answer_span = backend.prepare_inputs(
                sample.image, sample.question, greedy["raw_response"])
            if not full_inputs or not answer_span:
                print(f"skip {sid}: input preparation failed", flush=True)
                skipped += 1
                continue
            vgs = compute_vgs(backend, full_inputs, prompt_length, answer_span)
        except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as exc:
            print(f"skip {sid}: {type(exc).__name__}: {exc}", flush=True)
            skipped += 1
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            continue

        write_sample_json_line(args.output, run, {"sample": {"sample_id": sid}, "vgs": vgs.to_dict()})
        written += 1
        del full_inputs
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if written % 20 == 0:
            print(f"progress written={written} skipped={skipped}", flush=True)

    print(f"completed: written={written} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
