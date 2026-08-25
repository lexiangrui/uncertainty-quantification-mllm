#!/usr/bin/env python3
"""Shared loader joining generation, UQ and formal aligned-judge records.

The formal labels live in ``results/judging`` (protocol
``human-aligned-dual-judge-v1``: GPT-5.6-Terra + Gemini-3.7-Flash with human
adjudication of disagreements).  Exclusion rules follow ``results/metrics``:

- the aligned-judge file defines the label base (records with ``valid=false``
  are dropped);
- records without a UQ entry are kept with ``has_uq=False`` so callers can
  reproduce the "evaluated" set;
- a method score is ``None`` when that method reports ``valid=false``.

All statistics use the ``group_id``-clustered bootstrap implemented in
``src/evaluation.metrics`` so point estimates crosscheck ``results/metrics``
exactly.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import cluster_bootstrap_indices  # noqa: E402

MODELS = ("llava", "qwen", "internvl")
DATASETS = ("vilp", "hallusionbench", "mmvet")
METHODS = ("perplexity", "semantic_entropy", "umpire")
METHOD_SHORT = {"perplexity": "PPL", "semantic_entropy": "SE", "umpire": "UMPIRE"}

RESULTS = PROJECT_ROOT / "results"
OUT_ROOT = RESULTS / "analysis" / "exp1"

N_BOOT = 1000
SEED = 0


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_greedy(model: str, dataset: str) -> dict[str, dict]:
    """Greedy generation records keyed by sample_id."""
    out: dict[str, dict] = {}
    path = RESULTS / "generation" / model / "greedy" / f"{dataset}.jsonl"
    for row in iter_jsonl(path):
        if row.get("record_type") != "sample":
            continue
        sample, greedy = row["sample"], row.get("greedy", {})
        signals = greedy.get("signals") or {}
        out[sample["sample_id"]] = {
            "sample_id": sample["sample_id"],
            "group_id": sample.get("group_id") or sample["sample_id"],
            "dataset": dataset,
            "model": model,
            "question": sample.get("question"),
            "references": sample.get("references"),
            "metadata": sample.get("metadata") or {},
            "has_image": sample.get("has_image", True),
            "sections_valid": bool(greedy.get("sections_valid")),
            "answer": greedy.get("answer"),
            "vision": greedy.get("vision"),
            "reasoning": greedy.get("reasoning"),
            "token_count": signals.get("token_count"),
        }
    return out


def _empty_uq() -> dict:
    return {
        "has_uq": False,
        "scores": {m: None for m in METHODS},
        "se_clusters": None,
        "se_n_clusters": None,
        "se_dominant_mass": None,
        "ump_semantic_volume": None,
        "ump_incoherence_mean": None,
    }


def _parse_uq(uq: dict) -> dict:
    ppl = uq.get("perplexity", {})
    se = uq.get("semantic_entropy", {})
    ump = uq.get("umpire", {})
    clusters = se.get("clusters") if isinstance(se.get("clusters"), list) else None
    dominant = None
    if clusters:
        dominant = max((c.get("probability", 0.0) for c in clusters), default=None)
    return {
        "has_uq": True,
        "scores": {
            "perplexity": ppl.get("score") if ppl.get("valid") else None,
            "semantic_entropy": se.get("score") if se.get("valid") else None,
            "umpire": ump.get("score") if ump.get("valid") else None,
        },
        "se_clusters": clusters,
        "se_n_clusters": len(clusters) if clusters else None,
        "se_dominant_mass": dominant,
        "ump_semantic_volume": ump.get("semantic_volume") if ump.get("valid") else None,
        "ump_incoherence_mean": ump.get("incoherence_mean") if ump.get("valid") else None,
    }


def load_cell(model: str, dataset: str) -> list[dict]:
    """Judge-valid joined records for one model x dataset cell."""
    greedy = load_greedy(model, dataset)
    uq: dict[str, dict] = {}
    for row in iter_jsonl(RESULTS / "uq" / model / f"{dataset}.jsonl"):
        if row.get("record_type") != "sample":
            continue
        sid = row["sample"]["sample_id"]
        uq[sid] = _parse_uq(row.get("uq") or {})

    joined: list[dict] = []
    for row in iter_jsonl(RESULTS / "judging" / model / f"{dataset}.jsonl"):
        if row.get("record_type") != "sample":
            continue
        sample, judge = row["sample"], row.get("judge", {})
        sid = sample["sample_id"]
        base = greedy.get(sid)
        if base is None or not judge.get("valid"):
            continue
        rec = dict(base)
        rec["group_id"] = sample.get("group_id") or base["group_id"]
        rec.update(uq.get(sid) or _empty_uq())
        rec["correct"] = judge.get("correct")
        rec["hallucination"] = judge.get("hallucination")
        rec["rating"] = judge.get("rating")
        rec["hallucination_types"] = judge.get("hallucination_types") or []
        joined.append(rec)
    return joined


def load_all_cells() -> dict[tuple[str, str], list[dict]]:
    return {(m, d): load_cell(m, d) for m in MODELS for d in DATASETS}


def evaluated(records: list[dict], method: str | None = None) -> list[dict]:
    """Records with a UQ entry (and a valid score for ``method`` when given)."""
    out = []
    for rec in records:
        if not rec["has_uq"]:
            continue
        if method is not None and rec["scores"].get(method) is None:
            continue
        out.append(rec)
    return out


def all3_valid(records: list[dict]) -> list[dict]:
    return [r for r in records if r["has_uq"] and all(r["scores"].get(m) is not None for m in METHODS)]


def cell_percentiles(records: list[dict]) -> dict[str, float]:
    """Average-rank percentile of each method score within the given records."""
    out = {}
    for method in METHODS:
        values = np.array([r["scores"][method] for r in records], dtype=float)
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        sorted_vals = values[order]
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
                j += 1
            ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        out[method] = ranks / len(values)
    return out


def bootstrap_reps(records: list[dict], n_bootstrap: int = N_BOOT, seed: int = SEED):
    groups = [r["group_id"] for r in records]
    return cluster_bootstrap_indices(groups, n_bootstrap=n_bootstrap, seed=seed)


def percentile_ci(values: list) -> dict:
    defined = [v for v in values if v is not None]
    undefined = len(values) - len(defined)
    if not defined:
        return {"ci_low": None, "ci_high": None, "undefined_replicates": undefined}
    low, high = np.percentile(defined, (2.5, 97.5))
    return {"ci_low": float(low), "ci_high": float(high), "undefined_replicates": undefined}


def metric_ci(
    fn: Callable[[np.ndarray, np.ndarray], float | None],
    scores: np.ndarray,
    labels: np.ndarray,
    reps: list[np.ndarray],
) -> dict:
    values = [fn(scores[idx], labels[idx]) for idx in reps]
    ci = percentile_ci(values)
    return {"value": fn(scores, labels), **ci}


def write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def answer_format(answer: str | None) -> str:
    """Coarse answer-format bucket used by the LUH attribution module."""
    text = (answer or "").strip().lower()
    if not text:
        return "empty"
    if text in {"yes", "no"}:
        return "yes/no"
    stripped = text.strip(".,!?;:%$()[]{}")
    if stripped.replace(".", "").replace(",", "").isdigit():
        return "numeric"
    if len(text.split()) <= 3:
        return "short(<=3w)"
    return "long"


def fmt_ci(value: float | None, ci: dict | None = None) -> str:
    if value is None:
        return "N/A"
    if ci is None or ci.get("ci_low") is None:
        return f"{value:.3f}"
    return f"{value:.3f} ({ci['ci_low']:.3f}, {ci['ci_high']:.3f})"
