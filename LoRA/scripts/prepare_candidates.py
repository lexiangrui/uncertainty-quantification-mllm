#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lora_format.vqav2 import (  # noqa: E402
    assign_splits,
    collect_candidates,
    download_images,
    select_distinct_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select leakage-free VQAv2 train samples.")
    parser.add_argument("--questions-zip", type=Path, required=True)
    parser.add_argument("--annotations-zip", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=1600)
    parser.add_argument("--validation-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--min-agreement", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download-workers", type=int, default=12)
    parser.add_argument(
        "--image-base-url",
        default="http://images.cocodataset.org/train2014",
        help="Base URL used when downloading selected COCO train2014 images.",
    )
    parser.add_argument(
        "--exclude-candidates",
        type=Path,
        help="JSONL candidates whose question IDs and image IDs must not be selected again.",
    )
    parser.add_argument(
        "--candidate-file",
        default="candidates.jsonl",
        help="Name of the candidate JSONL written under --output-root.",
    )
    parser.add_argument(
        "--manifest-file",
        default="candidate_manifest.json",
        help="Name of the selection manifest written under --output-root.",
    )
    parser.add_argument("--skip-image-download", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    sizes = (args.train_size, args.validation_size, args.test_size)
    total = sum(sizes)
    if any(size < 0 for size in sizes) or total < 1:
        raise ValueError("split sizes must be non-negative and sum to a positive sample count")
    if args.min_agreement < 1 or args.download_workers < 1:
        raise ValueError("agreement and worker count must be positive")

    candidates = collect_candidates(args.questions_zip, args.annotations_zip, args.min_agreement)
    excluded_question_ids: set[int] = set()
    excluded_image_ids: set[int] = set()
    if args.exclude_candidates:
        existing = read_jsonl(args.exclude_candidates)
        excluded_question_ids = {int(row["question_id"]) for row in existing}
        excluded_image_ids = {int(row["image_id"]) for row in existing}
        candidates = [
            row
            for row in candidates
            if row["question_id"] not in excluded_question_ids and row["image_id"] not in excluded_image_ids
        ]
    selected = select_distinct_images(candidates, total, args.seed)
    selected = assign_splits(selected, *sizes)
    args.output_root.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_root / args.candidate_file
    with candidate_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if not args.skip_image_download:
        download_images(
            selected,
            args.output_root / "images",
            args.download_workers,
            args.image_base_url,
        )
    manifest = {
        "source": "VQAv2 train2014 official questions and annotations",
        "sample_count": total,
        "splits": {"train": sizes[0], "validation": sizes[1], "test": sizes[2]},
        "split_unit": "image_id (one selected question per image)",
        "selection": "round-robin over VQAv2 question_type",
        "min_human_answer_agreement": args.min_agreement,
        "seed": args.seed,
        "excluded_question_count": len(excluded_question_ids),
        "excluded_image_count": len(excluded_image_ids),
    }
    (args.output_root / args.manifest_file).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
