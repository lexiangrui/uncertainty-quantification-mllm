#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_jpeg(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(3) == b"\xff\xd8\xff"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=int)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.candidates.read_text().splitlines() if line]
    ids = [row["question_id"] for row in rows]
    if len(rows) != args.expected:
        raise ValueError(f"expected {args.expected} candidates, found {len(rows)}")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate question_id values are not unique")
    missing = [row["image_file"] for row in rows if not (args.image_dir / row["image_file"]).is_file()]
    if missing:
        raise ValueError(f"missing {len(missing)} images; first={missing[:3]}")
    non_jpeg = [
        row["image_file"]
        for row in rows
        if not is_jpeg(args.image_dir / row["image_file"])
    ]
    if non_jpeg:
        raise ValueError(f"non-JPEG content in {len(non_jpeg)} images; first={non_jpeg[:3]}")
    print(
        json.dumps(
            {
                "candidates": len(rows),
                "unique_question_ids": len(set(ids)),
                "unique_images": len({row["image_file"] for row in rows}),
                "missing_images": 0,
                "non_jpeg_images": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
