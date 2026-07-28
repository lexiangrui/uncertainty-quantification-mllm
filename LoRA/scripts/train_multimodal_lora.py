#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lora_format.multimodal_sft import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    args = parser.parse_args()
    train(
        json.loads(args.config.read_text()),
        args.max_train_samples,
        args.max_validation_samples,
        args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
