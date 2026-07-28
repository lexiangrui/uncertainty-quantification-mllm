"""Run five CVBench examples and report token-level top-k changes."""
import argparse
import json
import sys
from pathlib import Path

import torch

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import iter_samples
from model import LlavaMalpRunner
from perturb import PerturbSpec


def topk(scores, tokenizer, k):
    probs = torch.softmax(scores.float(), dim=-1)
    values, indices = probs.topk(k, dim=-1)
    rows = []
    for v, i in zip(values[0].tolist(), indices[0].tolist(), strict=True):
        rows.append({"token_id": i, "text": tokenizer.decode([i]), "prob": float(v)})
    return rows


def distribution_change(before, after):
    before_logp = torch.log_softmax(before.float(), dim=-1)
    after_logp = torch.log_softmax(after.float(), dim=-1)
    kl = (before_logp.exp() * (before_logp - after_logp)).sum(dim=-1).mean()
    return {"kl_before_after": float(kl.item())}


def teacher_forced_summary(runner, inputs, generated, k):
    """Score a generated answer without installing any perturbation hook."""
    teacher = runner.build_teacher_forcing_inputs(
        inputs, generated["answer_ids"], runner.build_answer_mask(generated["answer_ids"])
    )
    logits = runner.forward_original(teacher)["response_logits"]
    mask = teacher["answer_mask"]
    return {
        "nll": runner.mean_nll(logits, teacher["answer_ids"], mask),
        "tokens": [
            {
                "step": i,
                "target_token_id": int(teacher["answer_ids"][0, i].item()),
                "target_text": runner.processor.tokenizer.decode(
                    [int(teacher["answer_ids"][0, i].item())]
                ),
                "topk": topk(logits[:, i, :], runner.processor.tokenizer, k),
            }
            for i in range(logits.shape[1])
            if bool(mask[0, i])
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--output", type=Path, default=Path("results/malp/cvbench_final_hidden_topk.json"))
    args = ap.parse_args()
    runner = LlavaMalpRunner()
    records = []
    for sample in iter_samples("cvbench", split="test", limit=args.limit):
        inputs = runner.prepare_inputs(sample["image"], sample["question"])
        base = runner.greedy_generate(inputs)
        spec = PerturbSpec("joint", "final_hidden", "norm_isotropic", args.sigma, 1.0, args.seed)
        pert = runner.generate_with_perturbation(inputs, spec)
        n = min(len(base["scores"]), len(pert["scores"]))
        records.append({
            "id": sample["id"], "question": sample["question"], "choices": sample.get("choices"),
            "answer_index": sample.get("answer_index"), "before": base["text"], "after": pert["text"],
            "perturbation": {"stage": "final_hidden", "application": "prompt_prefill_last_token_once",
                             "mode": "norm_isotropic", "sigma": args.sigma, "seed": args.seed},
            "first_token_distribution_change": distribution_change(base["scores"][0], pert["scores"][0]),
            "tokens": [{"step": i, "before": topk(base["scores"][i], runner.processor.tokenizer, args.top_k),
                        "after": topk(pert["scores"][i], runner.processor.tokenizer, args.top_k)} for i in range(n)],
            "teacher_forcing_unperturbed": {
                "before_answer": teacher_forced_summary(runner, inputs, base, args.top_k),
                "after_answer": teacher_forced_summary(runner, inputs, pert, args.top_k),
            },
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
