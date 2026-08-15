#!/usr/bin/env python3
"""SAI extraction: vision logit lens reads + semantic anchor interventions.

Per sample (teacher-forced on the model's own greedy response):

* baseline forward with hidden states → static lens reading of every
  object's anchor token across ``--lens-layers``;
* for each intervention layer, object, sign and σ: rotate the top-k
  located visual states along the (centered, pre-norm) anchor direction and
  record Δ log-prob at *every* object's mention positions plus per-section
  ΔNLL — the full response matrix is stored for frozen-score post-processing;
* random-direction controls at the primary layer for specificity analysis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import iter_dataset
from src.utils import completed_sample_ids, load_jsonl_records, write_sample_json_line
from src.improvement.backend import SaiBackend
from src.improvement.sai import (
    InterventionHook,
    LensKit,
    SECTIONS,
    anchor_token_id,
    run_forward,
    section_spans,
    section_stats,
    stable_seed,
    unit_direction,
)


def load_objects(path: Path) -> dict[str, list[dict]]:
    """sample_id → object records (surface + mention char spans)."""
    out: dict[str, list[dict]] = {}
    if not path.exists():
        return out
    for obj in load_jsonl_records(path):
        sid = obj.get("sample", {}).get("sample_id")
        if sid:
            out[sid] = obj.get("objects", [])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy-input", required=True, type=Path)
    parser.add_argument("--objects-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--family", required=True,
                        choices=("llava_1_5", "qwen2_5_vl", "internvl3_5"))
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--dataset-source", required=True, type=Path)
    parser.add_argument("--lens-layers", type=int, nargs="+",
                        default=[8, 16, 20, 24, 28, 31])
    parser.add_argument("--intervene-layers", type=int, nargs="+",
                        default=[20, 28])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[2.0, 4.0])
    parser.add_argument("--k-locate", type=int, default=16)
    parser.add_argument("--k-locate-large", type=int, default=128)
    parser.add_argument("--anchor-modes", nargs="+",
                        default=["unembed", "mention_state"],
                        choices=["unembed", "mention_state"])
    parser.add_argument("--locate-modes", nargs="+",
                        default=["topk", "topk_large", "soft"],
                        choices=["topk", "topk_large", "soft", "all"])
    parser.add_argument("--max-objects", type=int, default=3)
    parser.add_argument("--random-controls", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-ids-file", type=Path, default=None)
    args = parser.parse_args()

    sample_filter = None
    if args.sample_ids_file:
        sample_filter = {l.strip() for l in args.sample_ids_file.open() if l.strip()}

    gen_rows = load_jsonl_records(args.greedy_input)
    gen_run = gen_rows[0]["run"]
    dataset = gen_run["dataset"]
    records = {r["sample"]["sample_id"]: r for r in gen_rows[1:]}
    objects_by_sid = load_objects(args.objects_input)

    run = {
        "sai_output_version": "v1",
        "greedy_input": str(args.greedy_input.resolve()),
        "objects_input": str(args.objects_input.resolve()),
        "greedy_run": {k: gen_run.get(k) for k in ("dataset", "model_id", "prompt_version")},
        "lens_layers": args.lens_layers,
        "intervene_layers": args.intervene_layers,
        "sigmas": args.sigmas,
        "k_locate": args.k_locate,
        "k_locate_large": args.k_locate_large,
        "anchor_modes": args.anchor_modes,
        "locate_modes": args.locate_modes,
        "max_objects": args.max_objects,
        "random_controls": args.random_controls,
        "anchor": "centered_prenorm_unembed",
    }
    completed = completed_sample_ids(args.output, run)

    backend = SaiBackend(args.family, args.model_path, adapter_path=args.adapter_path)
    backend._load()
    lens = LensKit(backend)
    layers = backend.decoder_layers()
    n_layers = len(layers)
    for L in args.lens_layers + args.intervene_layers:
        if not 0 < L < n_layers:
            raise SystemExit(f"layer {L} out of range (0,{n_layers})")

    hook = InterventionHook()
    written = skipped = 0
    for sample in iter_dataset(dataset, args.dataset_source, args.limit):
        sid = sample.sample_id
        if sid in completed or (sample_filter and sid not in sample_filter):
            continue
        record = records.get(sid)
        objects = objects_by_sid.get(sid, [])
        if not record or not objects or not sample.image:
            skipped += 1
            continue
        greedy = record.get("greedy", {})
        raw = greedy.get("raw_response")
        if not greedy.get("sections_valid") or not raw:
            skipped += 1
            continue
        objects = objects[: args.max_objects]
        try:
            batch = backend.prepare_inputs(sample.image, sample.question, raw)
            if batch is None:
                skipped += 1
                continue
            spans = section_spans(raw)
            if spans is None:
                skipped += 1
                continue

            # Resolve objects: anchor token + mention gen positions.
            obj_specs = []
            for obj in objects:
                tok_id = anchor_token_id(backend, obj["surface"])
                if tok_id is None:
                    continue
                positions = sorted(
                    {p for m in obj["mentions"] for p in batch.char_span_tokens(
                        m["char_start"], m["char_end"])}
                )
                if not positions:
                    continue
                obj_specs.append(
                    {"surface": obj["surface"], "token_id": tok_id, "positions": positions}
                )
            if not obj_specs:
                skipped += 1
                continue

            vis_pos = backend.visual_positions(batch.inputs)
            n_visual = int(vis_pos.numel())
            if n_visual == 0:
                skipped += 1
                continue

            # Section token positions for NLL stats.
            gen_by_section = {
                name: batch.char_span_tokens(*getattr(spans, name)) for name in SECTIONS
            }

            # Baseline forward with hidden states.
            base, hidden = run_forward(
                backend, batch, want_argmax=True, want_hidden=True
            )
            section_stats(base, batch, spans, gen_by_section, None)
            base_argmax = base.argmax_ids

            # Static lens reading per object per lens layer.
            lens_out = []
            base_lens = {}  # (layer, surface) → topk_mean logprob
            for L in args.lens_layers:
                logits = lens.visual_logits(hidden[L], vis_pos)
                logp = torch.log_softmax(logits.float(), dim=-1)
                for spec in obj_specs:
                    lp = logp[0, :, spec["token_id"]]
                    k = min(args.k_locate, n_visual)
                    topk = lp.topk(k).values
                    base_lens[(L, spec["surface"])] = float(topk.mean())
                    lens_out.append({
                        "layer": L, "surface": spec["surface"],
                        "token_id": spec["token_id"],
                        "max_logprob": float(lp.max()),
                        "topk_mean_logprob": float(topk.mean()),
                        "mean_logprob": float(lp.mean()),
                    })

            # Interventions.
            interventions = []
            for L in args.intervene_layers:
                layer_module = layers[L]
                h_L = hidden[L]  # (1, seq, D) entry of layer L
                # Lens logprobs of every object's anchor token per visual token.
                logits = lens.visual_logits(h_L, vis_pos)
                logp = torch.log_softmax(logits.float(), dim=-1)

                # Locate sets and per-mode directions per object.
                plans = []  # (anchor_name, locate_name, positions, direction, weights)
                for spec in obj_specs:
                    lp = logp[0, :, spec["token_id"]]
                    k = min(args.k_locate, n_visual)
                    lp_top_idx = lp.topk(k).indices
                    k2 = min(args.k_locate_large, n_visual)
                    lp_top_idx2 = lp.topk(k2).indices
                    # soft weights: evidence above mean, normalized to max 1
                    w = ((lp - lp.mean()) / (lp.max() - lp.mean()).clamp_min(1e-6)).clamp_min(0)
                    for anchor_mode in args.anchor_modes:
                        if anchor_mode == "unembed":
                            direction = lens.anchor_direction(
                                spec["token_id"], center=True, device=backend.device
                            )
                            for locate in args.locate_modes:
                                if locate == "topk":
                                    plans.append((spec["surface"], anchor_mode, locate,
                                                  vis_pos[lp_top_idx], direction, None))
                                elif locate == "topk_large":
                                    plans.append((spec["surface"], anchor_mode, locate,
                                                  vis_pos[lp_top_idx2], direction, None))
                                elif locate == "soft":
                                    plans.append((spec["surface"], anchor_mode, locate,
                                                  vis_pos, direction, w))
                                elif locate == "all":
                                    plans.append((spec["surface"], anchor_mode, locate,
                                                  vis_pos, direction, None))
                        elif anchor_mode == "mention_state":
                            first_pos = batch.absolute(spec["positions"][0])
                            anchor_point = h_L[0, first_pos].float()  # (D,)
                            for locate in args.locate_modes:
                                if locate == "topk":
                                    sel = vis_pos[lp_top_idx]
                                elif locate == "topk_large":
                                    sel = vis_pos[lp_top_idx2]
                                else:
                                    sel = vis_pos
                                hv = h_L[0].index_select(0, sel).float()
                                toward = anchor_point.view(1, -1) - hv  # (P, D)
                                plans.append((spec["surface"], anchor_mode, locate,
                                              sel, toward, None))
                # Random control directions at this layer.
                dim = h_L.shape[-1]
                for i in range(args.random_controls):
                    rand_dir = unit_direction(
                        dim, stable_seed(sid, L, "rand", i), backend.device
                    )
                    plans.append((f"rand{i}", "random", "topk",
                                  vis_pos[logp[0, :, obj_specs[0]["token_id"]].topk(
                                      min(args.k_locate, n_visual)).indices],
                                  rand_dir, None))

                for sign in (+1, -1):
                    for sigma in args.sigmas:
                        for surface, anchor_mode, locate, positions, direction, weights in plans:
                            eff_direction = direction if sign > 0 else (
                                -direction if direction.dim() == 1 else -direction
                            )
                            meas, hidden_iv = run_forward(
                                backend, batch, hook=hook, layer_module=layer_module,
                                positions=positions, direction=eff_direction,
                                sigma=sigma, weights=weights, want_argmax=True,
                                want_hidden=True,
                            )
                            section_stats(meas, batch, spans, gen_by_section, base_argmax)
                            # Post-intervention lens reading deltas per lens layer:
                            # does injected semantics propagate to later readings?
                            lens_dread = {}
                            for L2 in args.lens_layers:
                                if L2 <= L:
                                    continue
                                iv_logits = lens.visual_logits(hidden_iv[L2], vis_pos)
                                iv_logp = torch.log_softmax(iv_logits.float(), dim=-1)
                                dl = []
                                for spec in obj_specs:
                                    lp2 = iv_logp[0, :, spec["token_id"]]
                                    k2 = min(args.k_locate, n_visual)
                                    base_e = base_lens[(L2, spec["surface"])]
                                    dl.append(float(lp2.topk(k2).values.mean() - base_e))
                                lens_dread[str(L2)] = dl
                            del hidden_iv
                            import math as _math

                            def _finite(rows):
                                return [
                                    [v if _math.isfinite(v) else None for v in row]
                                    for row in rows
                                ]

                            interventions.append({
                                "layer": L, "anchor": surface, "anchor_mode": anchor_mode,
                                "locate": locate, "sign": sign, "sigma": sigma,
                                "n_positions": int(positions.numel()),
                                "mention_dlogp": _finite([
                                    [meas.token_logprobs[p] - base.token_logprobs[p]
                                     for p in spec["positions"]]
                                    for spec in obj_specs
                                ]),
                                "mention_dlogit": _finite([
                                    [meas.token_logits[p] - base.token_logits[p]
                                     for p in spec["positions"]]
                                    for spec in obj_specs
                                ]),
                                "section_dnll": {
                                    name_s: meas.section_nll[name_s] - base.section_nll[name_s]
                                    for name_s in SECTIONS
                                    if name_s in meas.section_nll and name_s in base.section_nll
                                },
                                "section_flip": meas.section_flip,
                                "lens_dread": lens_dread,
                            })
                            del meas

            result = {
                "n_visual": n_visual,
                "objects": [
                    {"surface": s["surface"], "token_id": s["token_id"],
                     "positions": s["positions"]}
                    for s in obj_specs
                ],
                "lens": lens_out,
                "baseline": {
                    "section_nll": base.section_nll,
                    "mention_logprob": [
                        [base.token_logprobs[p] for p in spec["positions"]]
                        for spec in obj_specs
                    ],
                    "mention_logit": [
                        [base.token_logits[p] for p in spec["positions"]]
                        for spec in obj_specs
                    ],
                },
                "interventions": interventions,
            }
            write_sample_json_line(args.output, run,
                                   {"sample": {"sample_id": sid}, "sai": result})
            written += 1
            del hidden, base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if written % 10 == 0:
                print(f"progress written={written} skipped={skipped}", flush=True)
        except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as exc:
            print(f"skip {sid}: {type(exc).__name__}: {exc}", flush=True)
            skipped += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    print(f"completed: written={written} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
