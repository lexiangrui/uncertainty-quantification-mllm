#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "LoRA" / "src"))

from lora_format.format_evaluation import evaluate_response  # noqa: E402
from src.generation.prompt import XML_LORA_PROMPT_SHA256, build_prompt  # noqa: E402


def read_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 200:
        raise ValueError(f"held-out test must contain exactly 200 rows, found {len(rows)}")
    return rows


def sample_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_existing(path: Path, run: dict) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["run"] != run:
            raise ValueError("existing held-out output uses a different run configuration")
        completed.add(record["sample_id"])
    return completed


def generated_records(processor, sequences: torch.Tensor, prompt_length: int, expected: str) -> list[dict]:
    eos_id = processor.tokenizer.eos_token_id
    generated = sequences[:, prompt_length:]
    records = []
    for token_ids in generated.tolist():
        stopped_on_eos = eos_id is not None and eos_id in token_ids
        if stopped_on_eos:
            token_ids = token_ids[: token_ids.index(eos_id) + 1]
        text = processor.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        records.append(
            {
                "raw_response": text,
                "generated_token_count": len(token_ids),
                "stopped_on_eos": stopped_on_eos,
                **evaluate_response(text, expected),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a LLaVA LoRA adapter on the fixed 200-row held-out split.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples != 10:
        raise ValueError("held-out comparison is fixed to 10 sampled responses")
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    from peft import PeftModel
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    rows = read_rows(args.test_jsonl)
    run = {
        "model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()),
        "test_jsonl": str(args.test_jsonl.resolve()),
        "prompt_sha256": XML_LORA_PROMPT_SHA256,
        "greedy": {"do_sample": False},
        "sampling": {"do_sample": True, "temperature": 1.0, "num_samples": 10},
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_existing(args.output, run)

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    base = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    model = PeftModel.from_pretrained(base, args.adapter_path, local_files_only=True).eval()
    device = next(model.parameters()).device

    for row in rows:
        sample_id = str(row["id"])
        if sample_id in completed:
            continue
        prompt = build_prompt(str(row["question"]), True)
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt.user}],
        }]
        rendered = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        with Image.open(args.image_dir / row["image_file"]) as source:
            image = source.convert("RGB")
            inputs = processor(text=rendered, images=image, return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        common = {
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
            "pad_token_id": processor.tokenizer.eos_token_id,
        }
        with torch.inference_mode():
            greedy_sequences = model.generate(**inputs, do_sample=False, **common)
            torch.manual_seed(sample_seed(args.seed, sample_id))
            torch.cuda.manual_seed_all(sample_seed(args.seed, sample_id))
            sampled_sequences = model.generate(
                **inputs,
                do_sample=True,
                temperature=1.0,
                num_return_sequences=args.num_samples,
                **common,
            )
        expected = str(row["answer"])
        record = {
            "run": run,
            "sample_id": sample_id,
            "question": row["question"],
            "expected_answer": expected,
            "greedy": generated_records(processor, greedy_sequences, prompt_length, expected)[0],
            "samples": generated_records(processor, sampled_sequences, prompt_length, expected),
        }
        append_jsonl(args.output, record)
        print(json.dumps({"sample_id": sample_id, "completed": len(completed) + 1}), flush=True)
        completed.add(sample_id)


if __name__ == "__main__":
    main()
