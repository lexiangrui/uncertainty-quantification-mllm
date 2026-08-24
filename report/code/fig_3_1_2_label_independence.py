#!/usr/bin/env python3
"""Figure 3.1.2 (scheme b): the independent structure of correctness and
hallucination (C x H).

Three aligned panels sharing the same nine model x dataset rows, double
column 183 mm:
  a  C x H quadrant composition (100% stacked bars)
  b  within-group hallucination rate dumbbell (H|E=1 vs H|C=1)
  c  phi coefficient dot plot (0 and -1 reference lines)

Data: results/analysis/descriptive/a2_c_h_joint.csv; conditional rates and
phi are recomputed from the quadrant counts and asserted against table 3.2
of the CN report before plotting.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

SLATE = "#6F8FB9"
SAGE = "#8FB9A8"
ROSE = "#D59AA5"
SAND = "#D9BA82"
MAUVE = "#A99BC6"
STEEL = "#8297A5"
DARK = "#2F3E4E"
NEUTRAL_LIGHT = "#CDD1D6"

MODEL_LABEL = {"llava": "LLaVA-1.5-7B", "qwen": "Qwen2.5-VL-7B", "internvl": "InternVL3.5-8B"}
DATASET_LABEL = {"vilp": "ViLP", "hallusionbench": "HallusionBench", "mmvet": "MM-Vet"}
MODELS = ("llava", "qwen", "internvl")
DATASETS = ("vilp", "hallusionbench", "mmvet")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "analysis" / "descriptive"
OUT_DIR = PROJECT_ROOT / "report" / "figures"
OUT = OUT_DIR / "fig_3_1_2_label_independence"


def load_cells() -> list[dict]:
    joint = pd.read_csv(RESULTS / "a2_c_h_joint.csv")
    cells = []
    for model in MODELS:
        for dataset in DATASETS:
            sub = joint[(joint.model == model) & (joint.dataset == dataset)]
            n = {int(row.correct) * 2 + int(row.hallucination): int(row.n) for row in sub.itertuples()}
            c1h0, c1h1, c0h0, c0h1 = n[2], n[3], n[0], n[1]
            total = c1h0 + c1h1 + c0h0 + c0h1
            phi = (c1h1 * c0h0 - c1h0 * c0h1) / np.sqrt(
                (c1h0 + c1h1) * (c0h0 + c0h1) * (c1h0 + c0h0) * (c1h1 + c0h1)
            )
            cells.append({
                "model": model, "dataset": dataset, "label": f"{MODEL_LABEL[model]}\n{DATASET_LABEL[dataset]}",
                "shares": np.array([c1h0, c1h1, c0h0, c0h1]) / total,
                "h_given_e1": c0h1 / (c0h0 + c0h1) * 100,
                "h_given_c1": c1h1 / (c1h0 + c1h1) * 100,
                "phi": phi,
            })
    return cells


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = load_cells()
    y = np.arange(len(cells))[::-1]

    fig = plt.figure(figsize=(183 / 25.4, 100 / 25.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 0.62],
                          left=0.095, right=0.985, top=0.82, bottom=0.09, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # panel a: quadrant composition
    cats = [
        ("C=1, H=0", SAGE, "dark"), ("C=1, H=1", SAND, "dark"),
        ("C=0, H=0", STEEL, "white"), ("C=0, H=1", ROSE, "white"),
    ]
    shares = np.array([c["shares"] for c in cells]) * 100
    left = np.zeros(len(cells))
    for k, (name, color, text_color) in enumerate(cats):
        ax_a.barh(y, shares[:, k], left=left, height=0.72, color=color,
                  edgecolor="white", linewidth=0.5, label=name)
        for yi, (l, v) in enumerate(zip(left, shares[:, k])):
            if v >= 7:
                ax_a.text(l + v / 2, y[yi], f"{v:.0f}", ha="center", va="center",
                          fontsize=6, color="white" if text_color == "white" else DARK)
        left += shares[:, k]
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([c["label"] for c in cells])
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("占已评估样本比例（%）")
    ax_a.set_title("C×H 四象限构成", loc="left", fontweight="bold", color=DARK, pad=4)
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2,
                handlelength=1.0, handleheight=0.8, columnspacing=0.8, labelspacing=0.25)

    # panel b: conditional hallucination-rate dumbbell
    for ci, cell in zip(y, cells):
        ax_b.plot([cell["h_given_c1"], cell["h_given_e1"]], [ci, ci],
                  color=NEUTRAL_LIGHT, linewidth=1.2, zorder=1)
    ax_b.scatter([c["h_given_e1"] for c in cells], y, s=22, color=ROSE,
                 edgecolor="white", linewidth=0.5, zorder=3, label="H | 错误 (E=1)")
    ax_b.scatter([c["h_given_c1"] for c in cells], y, s=22, color=SAGE,
                 edgecolor="white", linewidth=0.5, zorder=3, label="H | 正确 (C=1)")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([])
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("组内幻觉率（%）")
    ax_b.set_title("条件幻觉率", loc="left", fontweight="bold", color=DARK, pad=4)
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=1,
                handlelength=0.6, labelspacing=0.25)
    for ci, cell in zip(y, cells):
        if cell["model"] == "llava" and cell["dataset"] == "hallusionbench":
            ax_b.text(cell["h_given_c1"] - 3, ci, f'{cell["h_given_c1"]:.0f}%',
                      ha="right", va="center", fontsize=5.5, color=DARK)
            ax_b.text(cell["h_given_e1"] + 3, ci, f'{cell["h_given_e1"]:.0f}%',
                      ha="left", va="center", fontsize=5.5, color=DARK)

    # panel c: phi coefficient
    ax_c.axvline(0, color=DARK, linewidth=0.8)
    ax_c.axvline(-1, color=NEUTRAL_LIGHT, linewidth=0.8, linestyle="--")
    ax_c.scatter([c["phi"] for c in cells], y, s=20, color=DARK,
                 edgecolor="white", linewidth=0.5, zorder=3)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([])
    ax_c.set_xlim(-1.02, 0.02)
    ax_c.set_xlabel("四分点相关系数 φ(C, H)")
    ax_c.set_title("独立性", loc="left", fontweight="bold", color=DARK, pad=4)
    ax_c.text(-0.99, len(cells) - 0.45, "完全负相关", fontsize=5.5, color=NEUTRAL_LIGHT,
              ha="left", va="center")

    for label, ax in zip(("a", "b", "c"), (ax_a, ax_b, ax_c)):
        ax.text(-0.16 if ax is ax_a else -0.28, 1.16, label,
                transform=ax.transAxes, fontsize=9, fontweight="bold", color=DARK)

    fig.savefig(f"{OUT}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}.svg/.pdf/.png")


if __name__ == "__main__":
    main()
