#!/usr/bin/env python3
"""Compute ECA per-layer direct-attention masses via a single forward pass."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import iter_dataset
from src.utils import completed_sample_ids, load_jsonl_records, write_sample_json_line
from src.improvement import EcaBackend
from src.improvement.eca import compute_eca


def _load_generation(path):
    rows = load_jsonl_records(path)
    return rows[0]["run"], rows[1:]


def _greedy_token_ids(record: dict, generation_path: Path) -> list[int]:
    descriptor = record.get("generation_tokens") or {}
    relative = descriptor.get("path")
    if not isinstance(relative, str) or "greedy" not in descriptor.get("keys", []):
        raise ValueError("generation record has no greedy token-ID sidecar")
    sidecar = generation_path.parent / relative
    payload = torch.load(sidecar, map_location="cpu", weights_only=True)
    token_ids = payload.get("greedy")
    if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1:
        raise ValueError(f"invalid greedy token IDs: {sidecar}")
    return [int(value) for value in token_ids.tolist()]


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
    run = {
        "method": "eca",
        "token_alignment": "exact_generated_ids",
        "greedy_input": str(args.greedy_input.resolve()),
        "greedy_run": gen_run,
    }
    completed = completed_sample_ids(args.output, run)
    records_by_id = {}
    for record in records:
        sid = record.get("sample", {}).get("sample_id")
        if not isinstance(sid, str) or sid in records_by_id:
            raise ValueError(f"invalid or duplicate generation sample_id: {sid!r}")
        records_by_id[sid] = record

    backend = EcaBackend(args.family, args.model_path,
                         adapter_path=args.adapter_path,
                         attn_implementation=args.attn_implementation)
    backend._load()

    written = skipped = 0
    for sample in iter_dataset(dataset, args.dataset_source, args.limit):
        sid = sample.sample_id
        if sid in completed or (
            sample_ids_filter is not None and sid not in sample_ids_filter
        ):
            continue
        record = records_by_id.get(sid)
        if not record:
            print(f"skip {sid}: generation record missing", flush=True)
            skipped += 1
            continue
        greedy = record.get("greedy", {})
        if not greedy.get("sections_valid") or not greedy.get("raw_response") or not sample.image:
            skipped += 1
            continue
        token_ids = _greedy_token_ids(record, args.greedy_input)
        full_inputs, prompt_length, generated_buckets = backend.prepare_inputs_sections(
            sample.image, sample.question, greedy["raw_response"], token_ids
        )
        result = compute_eca(backend, full_inputs, prompt_length, generated_buckets)
        if result is None:
            print(f"skip {sid}: no decoder-layer masses", flush=True)
            skipped += 1
            continue

        write_sample_json_line(args.output, run, {"sample": {"sample_id": sid}, "eca": result.to_dict()})
        written += 1
        del full_inputs
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if written % 20 == 0:
            print(f"progress written={written} skipped={skipped}", flush=True)

    print(f"completed: written={written} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
