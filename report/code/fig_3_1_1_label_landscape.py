#!/usr/bin/env python3
"""Figure 3.1.1 (scheme a): accuracy-hallucination landscape per model x dataset.

Single panel, single column 89 mm. Nine cells plotted as points with 95%
bootstrap CI crossbars; models encoded by colour, datasets by marker.
Data: results/analysis/descriptive/a2_label_stats.csv (cross-checked against
table 3.1 of the CN report).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from pathlib import Path

# ── mandatory editable-text rules + CJK support ────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK SC", "Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

SLATE = "#6F8FB9"
SAGE = "#8FB9A8"
ROSE = "#D59AA5"
MAUVE = "#A99BC6"
DARK = "#2F3E4E"
NEUTRAL_LIGHT = "#CDD1D6"

MODEL_LABEL = {"llava": "LLaVA-1.5-7B", "qwen": "Qwen2.5-VL-7B", "internvl": "InternVL3.5-8B"}
MODEL_COLOR = {"llava": SLATE, "qwen": SAGE, "internvl": MAUVE}
DATASET_LABEL = {"vilp": "ViLP", "hallusionbench": "HallusionBench", "mmvet": "MM-Vet"}
DATASET_MARKER = {"vilp": "o", "hallusionbench": "s", "mmvet": "D"}
MODELS = ("llava", "qwen", "internvl")
DATASETS = ("vilp", "hallusionbench", "mmvet")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "analysis" / "descriptive"
OUT_DIR = PROJECT_ROOT / "report" / "figures"
OUT = OUT_DIR / "fig_3_1_1_label_landscape"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = pd.read_csv(RESULTS / "a2_label_stats.csv")

    fig, ax = plt.subplots(figsize=(89 / 25.4, 78 / 25.4))
    for model in MODELS:
        for dataset in DATASETS:
            row = stats[(stats.model == model) & (stats.dataset == dataset)].iloc[0]
            color = MODEL_COLOR[model]
            ax.errorbar(
                row.accuracy, row.hallu_rate,
                xerr=[[row.accuracy - row.acc_ci_low], [row.acc_ci_high - row.accuracy]],
                yerr=[[row.hallu_rate - row.hr_ci_low], [row.hr_ci_high - row.hallu_rate]],
                fmt="none", ecolor=color, elinewidth=0.8, capsize=1.5, capthick=0.8, alpha=0.75,
            )
            ax.scatter(
                row.accuracy, row.hallu_rate,
                s=30, marker=DATASET_MARKER[dataset], facecolor=color,
                edgecolor="white", linewidth=0.6, zorder=3,
            )
    ax.set_xlabel("准确率 Accuracy")
    ax.set_ylabel("幻觉率 Hallucination Rate")
    ax.set_xlim(0.15, 0.75)
    ax.set_ylim(0.10, 0.85)
    ax.grid(True, linewidth=0.4, color=NEUTRAL_LIGHT, alpha=0.55)
    ax.set_axisbelow(True)
    model_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6,
               markerfacecolor=MODEL_COLOR[m], markeredgecolor="white", markeredgewidth=0.8,
               label=MODEL_LABEL[m])
        for m in MODELS
    ]
    dataset_handles = [
        Line2D([0], [0], marker=DATASET_MARKER[d], linestyle="", markersize=6,
               markerfacecolor=NEUTRAL_LIGHT, markeredgecolor=DARK, markeredgewidth=0.8,
               label=DATASET_LABEL[d])
        for d in DATASETS
    ]
    ax.legend(handles=model_handles + dataset_handles, loc="upper right", ncol=2,
              handletextpad=0.2, columnspacing=0.7, borderaxespad=0.2, labelspacing=0.3)
    ax.set_title("准确率–幻觉率格局（误差线为 95% bootstrap 置信区间）",
                 loc="left", fontweight="bold", color=DARK, pad=6)

    fig.savefig(f"{OUT}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}.svg/.pdf/.png")


if __name__ == "__main__":
    main()
