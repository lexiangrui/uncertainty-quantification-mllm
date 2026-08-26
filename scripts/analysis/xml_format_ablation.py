#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ablation.xml_format import load_manifest, summarize_model, write_csv
from src.llm_judge.paths import judge_directory_name


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the paired XML-format ablation.")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=PROJECT_ROOT / "results/ablation/xml_format",
    )
    parser.add_argument("--judge-model", default="gemini-3.7-flash")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    manifest = load_manifest(args.experiment_root / "sample_manifest.json")
    judge_root = args.experiment_root / judge_directory_name(args.judge_model)
    metric_rows = []
    for model in ("llava", "qwen", "internvl"):
        _model_formats, model_metrics = summarize_model(
            model=model,
            manifest=manifest,
            generation_root=args.experiment_root / "generation",
            judge_root=judge_root,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        metric_rows.extend(model_metrics)
    analysis_root = args.experiment_root / "analysis"
    write_csv(analysis_root / "paired_performance.csv", metric_rows)
    by_model = {
        model: {
            row["metric"]: row for row in metric_rows if row["model"] == model
        }
        for model in ("llava", "qwen", "internvl")
    }
    lines = [
        "# XML 格式适配消融结果",
        "",
        f"> 两种条件均使用 `{args.judge_model}`；初始共享随机样本数：{manifest['sample_size']}；抽样种子：{manifest['seed']}。XML-LoRA 复用主实验 Gemini 标签，原生回答全部以 `raw_response` 原样提交。仅原生回答完整包含 Visual Observation、Reasoning 和 Final Answer 三段的样本进入各模型的配对统计。",
        "",
        "## 配对正确率与幻觉率",
        "",
        "| 模型 | 配对 n | XML 正确率 | 原生正确率 | 正确率差值（95% CI） | McNemar p | XML 幻觉率 | 原生幻觉率 | 幻觉率差值（95% CI） | McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("llava", "qwen", "internvl"):
        accuracy = by_model[model]["accuracy"]
        hallucination = by_model[model]["hallucination_rate"]
        lines.append(
            f"| {model} | {accuracy['paired_n']} | "
            f"{_fmt(accuracy['xml_lora_rate'])} | {_fmt(accuracy['native_prompt_rate'])} | "
            f"{_fmt(accuracy['delta_xml_minus_native'])} "
            f"({_fmt(accuracy['delta_ci_low'])}, {_fmt(accuracy['delta_ci_high'])}) | "
            f"{accuracy['mcnemar_exact_p']:.4g} | "
            f"{_fmt(hallucination['xml_lora_rate'])} | {_fmt(hallucination['native_prompt_rate'])} | "
            f"{_fmt(hallucination['delta_xml_minus_native'])} "
            f"({_fmt(hallucination['delta_ci_low'])}, {_fmt(hallucination['delta_ci_high'])}) | "
            f"{hallucination['mcnemar_exact_p']:.4g} |"
        )
    analysis_root.mkdir(parents=True, exist_ok=True)
    (analysis_root / "结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (analysis_root / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "xml-format-ablation-analysis-v3",
                "judge_model": args.judge_model,
                "sample_set_sha256": manifest["sample_set_sha256"],
                "pairing_filter": "native_prompt greedy.sections_valid=true and both judge labels valid",
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_seed": args.bootstrap_seed,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote XML-format ablation analysis to {analysis_root}")


if __name__ == "__main__":
    main()
