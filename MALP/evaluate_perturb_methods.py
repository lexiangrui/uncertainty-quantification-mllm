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

from io_utils import append_jsonl, load_jsonl_by_id
from judge.choice import LETTERS, RegexChoiceJudge, extract_choice_letter


LOGGER = logging.getLogger("malp.evaluate_perturb_methods")
_CHOICE_JUDGE = RegexChoiceJudge()


def roc_auc_score(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for score, label in zip(scores, labels, strict=True) if label == 1]
    negatives = [score for score, label in zip(scores, labels, strict=True) if label == 0]
    if not positives or not negatives:
        return None
    pairs = sorted(zip(scores, labels, strict=True), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += sum(average_rank for _, label in pairs[index:end] if label == 1)
        index = end
    return (rank_sum - len(positives) * (len(positives) + 1) / 2.0) / (
        len(positives) * len(negatives)
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def evaluate_record(record: dict, filter_modes: set[str] | None = None) -> list[dict]:
    nll0 = record["nll0"]
    groups = {(p["stage"], p["mode"]) for p in record["perturbations"]}
    if filter_modes is not None:
        groups = {group for group in groups if group[1] in filter_modes}

    choices = record.get("choices")
    answer_index = record.get("answer_index")
    prediction = record.get("prediction", "")
    if choices and answer_index is not None:
        prediction_choice = extract_choice_letter(prediction, len(choices))
        gold_choice = LETTERS[answer_index]
        correct = _CHOICE_JUDGE.judge(prediction, answer_index, choices, mode="letter")
    else:
        prediction_choice = gold_choice = correct = None

    expanded = []
    for stage, mode in sorted(groups):
        values = [
            p for p in record["perturbations"]
            if p["stage"] == stage and p["mode"] == mode
        ]
        seeds = [p["seed"] for p in values]
        if not values or len(set(seeds)) != len(seeds):
            raise ValueError(
                f"record {record['id']!r} stage={stage!r} mode={mode!r} has missing/duplicate seeds"
            )
        pis = mean([p["nll"] - nll0 for p in values])
        kl = mean([p["kl"] for p in values])
        expanded.append(
            {
                "id": record["id"],
                "dataset": record.get("dataset"),
                "stage": stage,
                "mode": mode,
                "layers": values[0].get("layers"),
                "prediction": prediction,
                "prediction_choice": prediction_choice,
                "gold_choice": gold_choice,
                "correct": correct,
                "error": None if correct is None else int(not correct),
                "nll0": nll0,
                "nll_mean": mean([p["nll"] for p in values]),
                "pis": pis,
                "kl": kl,
            }
        )
    return expanded


def summarize(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["stage"], record["mode"]), []).append(record)
    summaries = []
    for (stage, mode), items in sorted(grouped.items()):
        judged = [item for item in items if item["error"] is not None]
        labels = [item["error"] for item in judged]
        summary = {
            "stage": stage,
            "mode": mode,
            "layers": items[0].get("layers"),
            "num_samples": len(items),
            "accuracy": mean([1.0 if item["correct"] else 0.0 for item in judged]),
            "nll0_mean": mean([item["nll0"] for item in items]),
            "pis_mean": mean([item["pis"] for item in items]),
            "kl_mean": mean([item["kl"] for item in items]),
        }
        if labels and len(set(labels)) == 2:
            summary["auroc_nll0_error"] = roc_auc_score(labels, [item["nll0"] for item in judged])
            summary["auroc_pis_error"] = roc_auc_score(labels, [item["pis"] for item in judged])
            summary["auroc_kl_error"] = roc_auc_score(labels, [item["kl"] for item in judged])
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--modes", nargs="+", default=None)
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
                    "id=%s stage=%s mode=%s correct=%s pis=%.6f kl=%.6f",
                    item["id"], item["stage"], item["mode"], item["correct"], item["pis"], item["kl"]
                )
    summaries = summarize(expanded)
    summary_output = args.summary_output or args.output.with_suffix(".summary.json")
    summary_output.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
