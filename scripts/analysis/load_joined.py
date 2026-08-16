"""Shared loader joining generation, UQ and judge records for the analysis modules.

Exclusion rules follow results/metrics exactly:
- judge-invalid records are dropped;
- records without a UQ entry are kept with ``has_uq=False`` so callers can
  reproduce the "evaluated" set (judge valid AND UQ record present);
- a method score is None when that method is invalid for the sample.
"""

from __future__ import annotations

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

RESULTS = PROJECT_ROOT / "results"


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


def load_sample_answers(model: str, dataset: str) -> dict[str, list[str | None]]:
    """Final-answer strings of the K sampled responses, keyed by sample_id."""
    out: dict[str, list[str | None]] = {}
    path = RESULTS / "generation" / model / "samples" / f"{dataset}.jsonl"
    for row in iter_jsonl(path):
        if row.get("record_type") != "sample":
            continue
        sid = row["sample"]["sample_id"]
        out[sid] = [item.get("answer") for item in row.get("samples", [])]
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


def load_cell(model: str, dataset: str, *, with_sample_answers: bool = False) -> list[dict]:
    """Judge-valid joined records for one model x dataset cell."""
    greedy = load_greedy(model, dataset)
    uq: dict[str, dict] = {}
    for row in iter_jsonl(RESULTS / "uq" / model / f"{dataset}.jsonl"):
        if row.get("record_type") != "sample":
            continue
        sid = row["sample"]["sample_id"]
        uq[sid] = _parse_uq(row.get("uq") or {})
    answers = load_sample_answers(model, dataset) if with_sample_answers else {}

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
        if with_sample_answers:
            rec["sample_answers"] = answers.get(sid)
        joined.append(rec)
    return joined


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


def cluster_ci(
    stat: Callable[[np.ndarray], float | None],
    records: list[dict],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Percentile CI of ``stat`` over group_id-clustered bootstrap replicates."""
    groups = [r["group_id"] for r in records]
    replicates = cluster_bootstrap_indices(
        groups, n_bootstrap=n_bootstrap, seed=seed
    )
    values = [stat(idx) for idx in replicates]
    defined = [v for v in values if v is not None]
    undefined = len(values) - len(defined)
    if not defined:
        return {"ci_low": None, "ci_high": None, "undefined_replicates": undefined}
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(defined, (tail, 100.0 - tail))
    return {"ci_low": float(low), "ci_high": float(high), "undefined_replicates": undefined}


def write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_answer(text: str | None) -> str:
    """Lowercase and strip surrounding whitespace/punctuation for exact matching."""
    if not text:
        return ""
    return text.strip().strip(".,!?;:'\"()[]{}").strip().lower()
