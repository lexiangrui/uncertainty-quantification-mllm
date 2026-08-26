#!/usr/bin/env python3
"""Plot ERA-score distributions on the fixed LUH hard subsets."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "analysis" / "exp2" / "era_sample_scores.csv"
FIG_DIR = ROOT / "report" / "figures"
STEM = "fig_5_2_era_score_distribution"

MODELS = ["llava", "qwen", "internvl"]
MODEL_LABELS = ["LLaVA", "Qwen", "InternVL"]
GROUPS = ["non_hallucination", "LUH"]
GROUP_LABELS = {
    "non_hallucination": "Non-hallucination",
    "LUH": "LUH",
}
COLORS = {
    "non_hallucination": "#6F91B8",
    "LUH": "#D98178",
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.7,
            "grid.color": "#D9DEE7",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_and_validate() -> pd.DataFrame:
    data = pd.read_csv(INPUT)
    required = {"model", "group", "era", "sample_id"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if data["era"].isna().any() or not data["era"].between(0, 1).all():
        raise ValueError("ERA scores must be finite values in [0, 1]")
    counts = data.groupby(["model", "group"]).size().to_dict()
    expected = {(model, group): 200 for model in MODELS for group in GROUPS}
    if counts != expected:
        raise ValueError(f"unexpected model/group counts: {counts}")
    if data.duplicated(["model", "sample_id"]).any():
        raise ValueError("model/sample_id pairs must be unique")
    return data


def save_figure(fig: mpl.figure.Figure) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIG_DIR / f"{STEM}.{extension}",
            dpi=600 if extension == "png" else None,
            bbox_inches="tight",
            pad_inches=0.04,
        )
def plot(data: pd.DataFrame) -> mpl.figure.Figure:
    fig, ax = plt.subplots(figsize=(6.7, 3.15))
    centers = np.arange(len(MODELS), dtype=float)
    offsets = {"non_hallucination": -0.18, "LUH": 0.18}

    for group in GROUPS:
        values = [
            data.loc[(data["model"] == model) & (data["group"] == group), "era"].to_numpy()
            for model in MODELS
        ]
        positions = centers + offsets[group]
        violin = ax.violinplot(
            values,
            positions=positions,
            widths=0.31,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method="scott",
        )
        for body in violin["bodies"]:
            body.set_facecolor(COLORS[group])
            body.set_edgecolor("white")
            body.set_linewidth(0.7)
            body.set_alpha(0.88)

        box = ax.boxplot(
            values,
            positions=positions,
            widths=0.085,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            whis=1.5,
            boxprops={
                "facecolor": "white",
                "edgecolor": "#202833",
                "linewidth": 0.65,
                "alpha": 0.82,
            },
            whiskerprops={"color": "#202833", "linewidth": 0.65},
            capprops={"color": "#202833", "linewidth": 0.65},
            medianprops={"color": "#202833", "linewidth": 1.15},
        )
        for median in box["medians"]:
            median.set_zorder(4)

    ax.set_xticks(centers, MODEL_LABELS)
    ax.set_xlim(-0.55, len(MODELS) - 0.45)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Model")
    ax.set_ylabel("ERA score")
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)

    handles = [
        Patch(facecolor=COLORS[group], edgecolor="white", label=GROUP_LABELS[group])
        for group in GROUPS
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        columnspacing=1.5,
        handlelength=1.5,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.18, top=0.94)
    return fig


def main() -> None:
    configure_style()
    data = load_and_validate()
    fig = plot(data)
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
