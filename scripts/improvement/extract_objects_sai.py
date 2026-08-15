#!/usr/bin/env python3
"""Extract visual objects mentioned in greedy responses with an independent
text model (Qwen3-4B-Instruct-2507), then locate their mention char spans.

No dataset ground-truth object classes are used anywhere: objects come only
from the response text, and spans are located by exact surface matching.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import completed_sample_ids, load_jsonl_records, write_sample_json_line
from src.improvement.sai import SECTIONS, section_spans

MAX_MENTIONS = 4
MAX_OBJECTS = 6

_EXTRACT_PROMPT = """You are given a multimodal model's answer about an image, in three XML sections: <vision>, <reasoning>, <answer>.

List every concrete visual object that the answer text explicitly claims to be visible in the image.
"Visual objects" are physical things: animals, people, body parts, vehicles, furniture, tools, food, clothing, buildings, plants, natural objects, instruments, and any text/letters/digits/symbols said to be visible in the image.

Rules:
- Include an object only if the text asserts it is in the image (visible or depicted). Exclude objects from general knowledge, comparisons or hypotheticals.
- Use the exact word as written in the text (keep singular/plural as written).
- Do not include actions, verbs, attributes (colors, sizes, materials, counts), locations, or abstract concepts.
- Do not include "image" or "picture" itself.
- At most 8 objects, ordered by first appearance. If none, return an empty list.

Return only JSON, no other text: {{"objects": ["...", ...]}}

[Response]
{response}"""


def parse_objects(text: str) -> list[str]:
    """Parse the JSON object list from the text model's output."""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    items = obj.get("objects", []) if isinstance(obj, dict) else []
    out: list[str] = []
    for it in items:
        if isinstance(it, str):
            s = it.strip().strip(".,;:()")
            if s and len(s) <= 40 and s.lower() not in {"image", "picture", "photo"}:
                out.append(s)
    return out[:MAX_OBJECTS]


def locate_mentions(raw_response: str, surface: str, spans) -> list[dict]:
    """All case-insensitive whole-word occurrences of ``surface`` per section."""
    pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(surface) + r"(?![A-Za-z0-9])", re.IGNORECASE)
    mentions = []
    for m in pattern.finditer(raw_response):
        section = None
        for name in SECTIONS:
            s, e = getattr(spans, name)
            if m.start() >= s and m.end() <= e:
                section = name
                break
        if section is None:
            continue
        mentions.append(
            {"char_start": m.start(), "char_end": m.end(), "section": section}
        )
        if len(mentions) >= MAX_MENTIONS:
            break
    return mentions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-model-path", type=Path,
                        default=Path("/opt/lexiangrui/models/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-ids-file", type=Path, default=None)
    args = parser.parse_args()

    sample_filter = None
    if args.sample_ids_file:
        sample_filter = {l.strip() for l in args.sample_ids_file.open() if l.strip()}

    rows = load_jsonl_records(args.greedy_input)
    gen_run = rows[0]["run"]
    run = {
        "sai_objects_version": "v1",
        "greedy_input": str(args.greedy_input.resolve()),
        "text_model": str(args.text_model_path),
        "max_mentions": MAX_MENTIONS,
        "max_objects": MAX_OBJECTS,
    }
    completed = completed_sample_ids(args.output, run)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.text_model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.text_model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
    ).eval()

    written = skipped = 0
    for record in rows[1:]:
        sample = record.get("sample", {})
        sid = sample.get("sample_id")
        greedy = record.get("greedy", {})
        raw = greedy.get("raw_response")
        if not sid or sid in completed or (sample_filter and sid not in sample_filter):
            continue
        if not greedy.get("sections_valid") or not raw:
            print(f"skip {sid}: invalid response", flush=True)
            skipped += 1
            continue
        prompt = _EXTRACT_PROMPT.format(response=raw)
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        objects = parse_objects(reply)

        spans = section_spans(raw)
        obj_records = []
        for surface in objects:
            mentions = locate_mentions(raw, surface, spans) if spans else []
            if mentions:
                obj_records.append({"surface": surface, "mentions": mentions})
        write_sample_json_line(args.output, run, {
            "sample": {"sample_id": sid},
            "objects": obj_records,
            "raw_reply": reply[:400],
        })
        written += 1
        if written % 25 == 0:
            print(f"progress written={written} skipped={skipped}", flush=True)

    print(f"completed: written={written} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
