"""Evaluate uncertainty from latent-perturbed multiple-choice answers."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from choice_consistency import INVALID_CHOICE, choice_classes, consistency_metrics
from evaluate_perturb_methods import mean, roc_auc_score
from io_utils import append_jsonl, load_jsonl_by_id
from judge.choice import LETTERS, RegexChoiceJudge, extract_choice_letter


LOGGER = logging.getLogger("malp.evaluate_choice_consistency")
_CHOICE_JUDGE = RegexChoiceJudge()
SCORE_NAMES = (
    "answer_entropy",
    "variation_ratio",
    "base_flip_rate",
    "pairwise_disagreement",
)


def evaluate_record(record: dict, filter_modes: set[str] | None = None) -> list[dict]:
    choices = record.get("choices")
    answer_index = record.get("answer_index")
    if not choices or answer_index is None:
        raise ValueError(f"record {record.get('id')!r} is not multiple choice")
    generations = record.get("generations", [])
    groups = {(item["stage"], item["mode"]) for item in generations}
    if filter_modes is not None:
        groups = {group for group in groups if group[1] in filter_modes}

    prediction = record.get("prediction", "")
    base_choice = extract_choice_letter(prediction, len(choices)) or INVALID_CHOICE
    correct = _CHOICE_JUDGE.judge(prediction, answer_index, choices, mode="letter")
    expanded = []
    for stage, mode in sorted(groups):
        values = [
            item
            for item in generations
            if item["stage"] == stage and item["mode"] == mode
        ]
        seeds = [item["seed"] for item in values]
        if len(values) < 2 or len(set(seeds)) != len(seeds):
            raise ValueError(
                f"record {record['id']!r} stage={stage!r} mode={mode!r} "
                "has fewer than two generations or duplicate seeds"
            )
        classes = choice_classes([item["text"] for item in values], len(choices))
        metrics = consistency_metrics(classes, base_choice)
        expanded.append(
            {
                "id": record["id"],
                "dataset": record.get("dataset"),
                "stage": stage,
                "mode": mode,
                "layers": values[0].get("layers"),
                "sigma": values[0].get("sigma"),
                "num_generations": len(values),
                "prediction": prediction,
                "base_choice": base_choice,
                "gold_choice": LETTERS[answer_index],
                "correct": correct,
                "error": int(not correct),
                "answer_classes": classes,
                **metrics,
            }
        )
    return expanded


def summarize(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, float | None], list[dict]] = {}
    for record in records:
        key = (record["stage"], record["mode"], record.get("sigma"))
        grouped.setdefault(key, []).append(record)
    summaries = []
    for (stage, mode, sigma), items in sorted(grouped.items()):
        labels = [item["error"] for item in items]
        summary = {
            "stage": stage,
            "mode": mode,
            "sigma": sigma,
            "layers": items[0].get("layers"),
            "num_samples": len(items),
            "num_generations": items[0]["num_generations"],
            "accuracy": mean([1.0 if item["correct"] else 0.0 for item in items]),
            "invalid_rate_mean": mean([item["invalid_rate"] for item in items]),
        }
        for score_name in SCORE_NAMES:
            summary[f"{score_name}_mean"] = mean(
                [item[score_name] for item in items]
            )
            if len(set(labels)) == 2:
                summary[f"auroc_{score_name}_error"] = roc_auc_score(
                    labels, [item[score_name] for item in items]
                )
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--modes", nargs="+", choices=["norm_isotropic", "directional"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    records = list(load_jsonl_by_id(args.input).values())
    if not records:
        raise RuntimeError(f"no records found in {args.input}")
    filter_modes = set(args.modes) if args.modes else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    expanded = []
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            for item in evaluate_record(record, filter_modes):
                append_jsonl(handle, item)
                expanded.append(item)
                LOGGER.info(
                    "id=%s stage=%s mode=%s correct=%s entropy=%.6f flip=%.6f",
                    item["id"],
                    item["stage"],
                    item["mode"],
                    item["correct"],
                    item["answer_entropy"],
                    item["base_flip_rate"],
                )
    summaries = summarize(expanded)
    summary_output = args.summary_output or args.output.with_suffix(".summary.json")
    summary_output.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
