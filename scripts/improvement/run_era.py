#!/usr/bin/env python3
"""Compute ERA (Early Rationale Attribution) per-layer direct-attention masses via a single forward pass."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import iter_dataset
from src.improvement.backend import EraBackend
from src.improvement.era import compute_era
from src.utils import completed_sample_ids, load_jsonl_records, write_sample_json_line


def _load_generation(path: Path):
    rows = load_jsonl_records(path)
    return rows[0]["run"], rows[1:]


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


def _pending_records(
    records: list[dict],
    completed: set[str],
    sample_ids_filter: set[str] | None,
) -> dict[str, dict]:
    records_by_id: dict[str, dict] = {}
    for record in records:
        sid = record.get("sample", {}).get("sample_id")
        if not isinstance(sid, str) or sid in records_by_id:
            raise ValueError(f"invalid or duplicate generation sample_id: {sid!r}")
        records_by_id[sid] = record
    return {
        sid: record
        for sid, record in records_by_id.items()
        if sid not in completed
        and (sample_ids_filter is None or sid in sample_ids_filter)
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ERA (Early Rationale Attribution) component extraction")
    parser.add_argument("--greedy-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--family",
        required=True,
        choices=("llava_1_5", "qwen2_5_vl", "internvl3_5", "internvl3_5_original"),
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--dataset-source", required=True, type=Path)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-ids-file", type=Path, default=None)
    args = parser.parse_args()

    sample_ids_filter = _load_sample_ids(args.sample_ids_file)
    if sample_ids_filter is not None:
        print(f"Filtering to {len(sample_ids_filter)} IDs")

    gen_run, records = _load_generation(args.greedy_input)
    dataset = gen_run["dataset"]
    run = {
        "method": "era",
        "token_alignment": "continuous_slices_with_xml",
        "model_family": args.family,
        "model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else None,
        "greedy_input": str(args.greedy_input.resolve()),
        "greedy_run": gen_run,
    }
    if args.family == "internvl3_5_original":
        run["image_preprocessing"] = "vllm_internvl_dynamic_tiles_v1"
    if sample_ids_filter is not None:
        run["sample_filter"] = sorted(sample_ids_filter)
    completed = completed_sample_ids(args.output, run)
    records_by_id = _pending_records(records, completed, sample_ids_filter)
    if not records_by_id:
        print(f"completed: written=0 skipped={len(completed)} output={args.output}")
        return
    print(
        f"ERA pending attention forwards={len(records_by_id)} "
        f"completed={len(completed)}",
        flush=True,
    )

    backend = EraBackend(
        args.family,
        args.model_path,
        adapter_path=args.adapter_path,
        attn_implementation=args.attn_implementation,
    )
    backend._load()

    written = skipped = 0
    for sample in iter_dataset(dataset, args.dataset_source, args.limit):
        sid = sample.sample_id
        if sid not in records_by_id:
            continue
        record = records_by_id[sid]
        greedy = record.get("greedy", {})
        if not greedy.get("sections_valid") or not greedy.get("raw_response") or not sample.image:
            skipped += 1
            continue
        try:
            token_ids = _greedy_token_ids(record, args.greedy_input)
            full_inputs, prompt_length, generated_buckets = backend.prepare_inputs_sections(
                sample.image, sample.question, greedy["raw_response"], token_ids
            )
        except (ValueError, FileNotFoundError) as error:
            print(f"skip {sid}: {error}", flush=True)
            skipped += 1
            continue

        result = compute_era(backend, full_inputs, prompt_length, generated_buckets)
        if result is None:
            print(f"skip {sid}: no decoder-layer masses", flush=True)
            skipped += 1
            del full_inputs
            continue

        payload = result.to_dict()
        write_sample_json_line(
            args.output,
            run,
            {"sample": {"sample_id": sid}, "era": payload},
        )
        written += 1
        del full_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if written % 20 == 0:
            print(f"progress written={written} skipped={skipped}", flush=True)

    print(f"completed: written={written} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
