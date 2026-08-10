#!/usr/bin/env python3
"""Apply LLM-as-judge labels to an existing VAUQ JSONL result file.

This is the preferred cluster workflow for free-form datasets: run VAUQ on GPU
nodes first, then label the saved predictions from a networked login shell when
an API key is available.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
SRC = ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.llm_judge import OpenSourceJudge
from vauq.eval import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relabel VAUQ JSONL with a free-form judge.")
    parser.add_argument("--input", required=True, help="Input VAUQ JSONL file.")
    parser.add_argument("--output", default=None, help="Output JSONL. Default: <input>.llm_judged.jsonl")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--judge", choices=["open_source"], default="open_source")
    parser.add_argument("--resume", action="store_true", help="Append and skip IDs already in output.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls.")
    parser.add_argument("--model", default=None, help="Override QWEN_JUDGE_MODEL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_suffix(".llm_judged.jsonl")
    summary_path = Path(args.summary_output) if args.summary_output else out_path.with_suffix(".summary.json")

    rows = _read_jsonl(in_path)
    limit = None if args.limit is None or args.limit <= 0 else args.limit
    end = len(rows) if limit is None else min(len(rows), args.start + limit)
    selected = rows[args.start:end]

    done = _load_done(out_path) if args.resume else {}
    mode = "a" if args.resume and done else "w"
    model_path = args.model or os.environ["QWEN_JUDGE_MODEL"]
    judge = OpenSourceJudge(model_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open(mode, encoding="utf-8") as f:
        for row in tqdm(selected, desc="LLM judge"):
            row_id = str(row.get("id"))
            if row_id in done:
                continue
            labeled = dict(row)
            if "correct" in labeled and "correct_before_llm" not in labeled:
                labeled["correct_before_llm"] = labeled["correct"]
            correct = judge.judge(
                labeled.get("question", ""),
                [str(labeled.get("gt_ans"))],
                labeled.get("prediction", ""),
            )
            labeled["correct"] = correct
            labeled["judge"] = args.judge
            labeled["judge_result"] = getattr(judge, "last_result", None)
            f.write(json.dumps(labeled, ensure_ascii=False) + "\n")
            f.flush()
            if args.sleep > 0:
                time.sleep(args.sleep)

    records = _read_jsonl(out_path)
    summary = _summarize(records)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {str(row.get("id")): row for row in _read_jsonl(path)}


def _summarize(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    summary = _summarize_group(records)
    summary["n"] = len(records)
    groups: dict[str, list[dict]] = {}
    for row in records:
        groups.setdefault(row.get("subset") or "all", []).append(row)
    if len(groups) > 1:
        summary["per_subset"] = {name: _summarize_group(rs) for name, rs in sorted(groups.items())}
    return summary


def _summarize_group(records: list[dict]) -> dict:
    labeled = [r for r in records if r.get("correct") is not None]
    labels = [int(r["correct"]) for r in labeled]
    vauq = [r["scores"]["vauq"] for r in labeled]
    entropy = [r["scores"]["entropy"] for r in labeled]
    is_score = [r["scores"]["is_score"] for r in labeled]
    metrics = compute_metrics(labels, vauq, entropy, is_score)
    return {
        "n": len(records),
        "n_labeled": len(labels),
        "accuracy": metrics["accuracy"] if labels else float("nan"),
        "metrics": metrics["metrics"],
    }


if __name__ == "__main__":
    main()
