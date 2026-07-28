#!/usr/bin/env python3
"""Extract selected COCO2014 images from Hugging Face Parquet shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract selected COCO2014 images from Parquet shards.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    return parser.parse_args()


def target_files(path: Path) -> set[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["image_file"]) for row in rows}


def main() -> None:
    args = parse_args()
    wanted = target_files(args.candidates)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    missing = {name for name in wanted if not (args.image_dir / name).is_file()}
    shards = sorted(args.shard_dir.glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no train Parquet shards found in {args.shard_dir}")

    for shard in shards:
        if not missing:
            break
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(columns=["image_id", "image"], batch_size=512):
            ids = batch.column("image_id").to_pylist()
            images = batch.column("image").to_pylist()
            for image_id, image in zip(ids, images):
                filename = f"COCO_train2014_{int(image_id):012d}.jpg"
                if filename in missing:
                    image_bytes = image.get("bytes") if image else None
                    if not image_bytes:
                        raise ValueError(f"missing image bytes for {filename} in {shard}")
                    (args.image_dir / filename).write_bytes(image_bytes)
                    missing.remove(filename)
        print(json.dumps({"shard": shard.name, "remaining": len(missing)}), flush=True)

    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise RuntimeError(f"could not extract {len(missing)} requested images, including {preview}")
    print(json.dumps({"extracted": len(wanted), "status": "complete"}))


if __name__ == "__main__":
    main()
