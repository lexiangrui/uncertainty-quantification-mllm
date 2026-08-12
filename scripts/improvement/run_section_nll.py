#!/usr/bin/env python3
"""Compute per-section (vision/reasoning/answer) NLL from a teacher-forcing forward pass.

The hallucination label H judges the vision/reasoning part, not the answer.
Baseline PPL only uses answer NLL.  This script also computes vision_nll and
reasoning_nll to test whether the non-answer sections carry signal for
hallucination detection on the LUH subset.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import iter_dataset
from src.utils import completed_sample_ids, load_jsonl_records, write_sample_json_line
from src.improvement import LacBackend


def _section_spans(raw_response: str, tokenizer) -> dict[str, tuple[int, int]]:
    """Find token spans for vision, reasoning, answer sections."""
    # Find XML tag character positions
    pattern = r"<(vision|reasoning|answer)>(.*?)</\1>"
    matches = list(re.finditer(pattern, raw_response, re.DOTALL))
    if len(matches) < 3:
        return {}

    spans = {}
    for m in matches:
        tag = m.group(1)
        content_start = m.start(2)
        content_end = m.end(2)

        # Map character positions to token positions
        prefix = raw_response[:content_start]
        full = raw_response[:content_end]
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        full_tokens = tokenizer.encode(full, add_special_tokens=False)
        spans[tag] = (len(prefix_tokens), len(full_tokens))

    return spans


def _load_generation(path: Path) -> tuple[dict, list[dict]]:
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
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-ids-file", type=Path, default=None)
    args = parser.parse_args()

    sample_ids_filter = None
    if args.sample_ids_file is not None:
        sample_ids_filter = set()
        with args.sample_ids_file.open() as f:
            for line in f:
                sid = line.strip()
                if sid:
                    sample_ids_filter.add(sid)
        print(f"Filtering to {len(sample_ids_filter)} IDs")

    generation_run, records = _load_generation(args.greedy_input)
    dataset = generation_run["dataset"]

    run = {
        "section_nll_version": "v1",
        "greedy_input": str(args.greedy_input.resolve()),
        "greedy_run": generation_run,
    }
    completed = completed_sample_ids(args.output, run)

    backend = LacBackend(args.family, args.model_path,
                          adapter_path=args.adapter_path,
                          attn_implementation=args.attn_implementation)
    backend._load()
    tok = getattr(backend.processor, "tokenizer", backend.processor)

    written = skipped = 0
    for sample in iter_dataset(dataset, args.dataset_source, args.limit):
        sid = sample.sample_id
        if sid in completed or (sample_ids_filter and sid not in sample_ids_filter):
            continue
        record = next((r for r in records if r.get("sample", {}).get("sample_id") == sid), None)
        if not record:
            skipped += 1; continue
        greedy = record.get("greedy", {})
        raw = greedy.get("raw_response", "")
        if not greedy.get("sections_valid") or not raw or not sample.image:
            skipped += 1; continue

        try:
            full_inputs, prompt_length, _ = backend.prepare_inputs(sample.image, sample.question, raw)
            if full_inputs is None:
                skipped += 1; continue

            with torch.inference_mode():
                outputs = backend.model(**full_inputs, use_cache=False)

            logits = outputs.logits  # (1, seq, vocab)
            input_ids = full_inputs["input_ids"]

            # Get section spans within generated tokens
            gen_token_ids = tok.encode(raw, add_special_tokens=False)
            section_spans = _section_spans(raw, tok)

            result = {"valid": True}
            for section in ("vision", "reasoning", "answer"):
                if section not in section_spans:
                    result[section + "_nll"] = None
                    continue
                tok_start, tok_end = section_spans[section]
                abs_start = prompt_length + tok_start
                abs_end = prompt_length + tok_end
                if abs_end <= abs_start:
                    result[section + "_nll"] = None
                    result[section + "_token_count"] = 0
                    continue
                sec_logits = logits[:, abs_start - 1:abs_end - 1, :].float()
                sec_ids = input_ids[:, abs_start:abs_end].to(sec_logits.device)
                lp = F.log_softmax(sec_logits, dim=-1)
                tlp = lp.gather(-1, sec_ids.unsqueeze(-1)).squeeze(-1)
                result[section + "_nll"] = -tlp.mean().item()
                result[section + "_token_count"] = int(abs_end - abs_start)

            # Also compute max and std of per-token NLL within answer
            if "answer" in section_spans:
                tok_start, tok_end = section_spans["answer"]
                abs_start = prompt_length + tok_start
                abs_end = prompt_length + tok_end
                if abs_end > abs_start:
                    sec_logits = logits[:, abs_start - 1:abs_end - 1, :].float()
                    sec_ids = input_ids[:, abs_start:abs_end].to(sec_logits.device)
                    lp = F.log_softmax(sec_logits, dim=-1)
                    tlp = lp.gather(-1, sec_ids.unsqueeze(-1)).squeeze(-1)
                    per_token_nll = -tlp.cpu().numpy()
                    result["answer_nll_max"] = float(per_token_nll.max())
                    result["answer_nll_std"] = float(per_token_nll.std())

            del full_inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except (RuntimeError, ValueError):
            skipped += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        write_sample_json_line(args.output, run, {"sample": {"sample_id": sid}, "section_nll": result})
        written += 1
        if written % 20 == 0:
            print(f"progress written={written} skipped={skipped}", flush=True)

    print(f"completed: written={written} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
