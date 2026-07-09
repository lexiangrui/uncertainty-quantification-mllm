#!/usr/bin/env python3
"""Visualize Grad-VAUQ visual-token attributions for selected CVBench cases."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "baseline" / "vauq-repro" / "src"
GRAD_ROOT = ROOT / "Grad"
for path in (SRC, GRAD_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from grad_vauq.backends import build_backend
from grad_vauq.scoring import compute_grad_vauq_scores
from vauq.benchmarks import build_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=os.environ.get("LLAVA_7B_HF_MODEL"))
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--case-ids", nargs="+", type=int, required=True)
    parser.add_argument("--result-jsonl", default="results/grad_cvbench_llava-1.5-7b-hf_sdpa_zero_full.jsonl")
    parser.add_argument("--output-dir", default="results/grad_visualizations")
    parser.add_argument("--topk-ratio", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=1.2)
    parser.add_argument("--ablation-baseline", choices=["zero", "mean"], default="zero")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def load_rows(path: Path) -> dict[int, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row["id"])] = row
    return rows


def normalize_grid(scores: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(scores, [5, 99])
    if hi <= lo:
        hi = float(scores.max())
        lo = float(scores.min())
    norm = (scores - lo) / max(hi - lo, 1e-8)
    return np.clip(norm, 0.0, 1.0)


def heat_color(norm: np.ndarray) -> np.ndarray:
    red = (255 * norm).astype(np.uint8)
    green = (210 * np.clip(1.0 - np.abs(norm - 0.55) * 1.8, 0, 1)).astype(np.uint8)
    blue = (255 * (1.0 - norm)).astype(np.uint8)
    alpha = (70 + 150 * norm).astype(np.uint8)
    return np.stack([red, green, blue, alpha], axis=-1)


def make_overlay(image: Image.Image, score_grid: np.ndarray, selected: list[int]) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    norm = normalize_grid(score_grid)
    rgba = Image.fromarray(heat_color(norm), mode="RGBA").resize((width, height), Image.Resampling.BILINEAR)
    overlaid = Image.alpha_composite(image.convert("RGBA"), rgba)

    draw = ImageDraw.Draw(overlaid)
    gh, gw = score_grid.shape
    cell_w = width / gw
    cell_h = height / gh
    for idx in selected:
        y, x = divmod(idx, gw)
        draw.rectangle(
            [x * cell_w, y * cell_h, (x + 1) * cell_w, (y + 1) * cell_h],
            outline=(255, 20, 20, 170),
            width=max(1, int(min(cell_w, cell_h) / 7)),
        )
    return overlaid.convert("RGB")


def draw_text_block(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, width_chars: int, fill=(30, 30, 30)):
    font = ImageFont.load_default()
    x, y = xy
    for line in text.splitlines():
        for wrapped in textwrap.wrap(line, width=width_chars) or [""]:
            draw.text((x, y), wrapped, font=font, fill=fill)
            y += 15
    return y


def make_case_panel(case_id: int, sample: dict, full_row: dict, fresh_row: dict, output_dir: Path) -> Path:
    image = sample["img"].convert("RGB")
    score_grid = np.array(fresh_row["visual_scores"], dtype=np.float32).reshape(fresh_row["spatial_shape"])
    overlay = make_overlay(image, score_grid, fresh_row["selected_indices"])

    panel_w = 980
    image_w = 360
    image_h = int(image.height * image_w / image.width)
    original = image.resize((image_w, image_h), Image.Resampling.LANCZOS)
    heat = overlay.resize((image_w, image_h), Image.Resampling.LANCZOS)
    text_h = 250
    panel_h = max(image_h, 360) + text_h
    panel = Image.new("RGB", (panel_w, panel_h), "white")
    panel.paste(original, (20, 80))
    panel.paste(heat, (400, 80))

    draw = ImageDraw.Draw(panel)
    draw.text((20, 20), f"Case {case_id} | subset={full_row['subset']} | correct={full_row['correct']}", fill=(0, 0, 0))
    draw.text((20, 55), "Original", fill=(0, 0, 0))
    draw.text((400, 55), "Grad heatmap + selected top-k patches", fill=(0, 0, 0))

    metrics = full_row["scores"]
    meta = (
        f"GT: {full_row['gt_ans']}    Pred: {full_row['prediction']}\n"
        f"entropy={metrics['entropy']:.3f}, entropy_masked={metrics['entropy_masked']:.3f}, "
        f"IS={metrics['is_score']:.3f}, VAUQ={metrics['vauq']:.3f}\n"
        f"Question: {sample['question']}"
    )
    draw_text_block(draw, (20, max(image_h, 360) + 105), meta, width_chars=135)

    out = output_dir / f"case_{case_id}.png"
    panel.save(out)
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.result_jsonl))
    benchmark = build_benchmark("cvbench")
    backend = build_backend(
        "llava",
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
        adapter="llava",
    )

    panels = []
    details = []
    for case_id in args.case_ids:
        sample = benchmark.retrieve(case_id)
        answer, generated_ids = backend.generate_with_ids(
            sample["img"],
            sample["question"],
            temp=0.0,
            max_new_tokens=args.max_new_tokens,
        )
        result = compute_grad_vauq_scores(
            backend,
            sample["img"],
            sample["question"],
            generated_ids,
            topk_ratio=args.topk_ratio,
            alpha=args.alpha,
            selector_name="grad_x_act",
            ablation_baseline=args.ablation_baseline,
            answer=answer,
            store_visual_scores=True,
        )
        fresh = {
            "answer": result.answer,
            "selected_indices": result.selected_indices,
            "visual_scores": result.visual_scores,
            "spatial_shape": result.spatial_shape,
        }
        panel_path = make_case_panel(case_id, sample, rows[case_id], fresh, out_dir)
        panels.append(Image.open(panel_path).convert("RGB"))
        details.append(
            {
                "id": case_id,
                "subset": rows[case_id]["subset"],
                "correct": rows[case_id]["correct"],
                "gt": rows[case_id]["gt_ans"],
                "prediction": rows[case_id]["prediction"],
                "scores": rows[case_id]["scores"],
                "panel": str(panel_path),
            }
        )

    if panels:
        sheet_w = max(p.width for p in panels)
        sheet_h = sum(p.height for p in panels)
        sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
        y = 0
        for panel in panels:
            sheet.paste(panel, (0, y))
            y += panel.height
        sheet.save(out_dir / "contact_sheet.png")

    (out_dir / "cases.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
