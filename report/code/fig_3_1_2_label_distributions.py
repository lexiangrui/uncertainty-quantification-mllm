"""Generate the two descriptive-statistics figures used in report section 3.1.2.

The source counts are the audited values in Tables 3.1.2a and 3.1.2b of the
Chinese experiment report.  The script validates every marginal total before
drawing so a future table edit cannot silently produce an inconsistent figure.
"""

from pathlib import Path
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "report" / "figures"

MODELS = ["LLaVA", "Qwen", "InternVL"]
JOINT_LABELS = [
    "Correct, no hallucination",
    "Correct, hallucination",
    "Incorrect, no hallucination",
    "Incorrect, hallucination",
]
TYPE_LABELS = ["Vision-only", "Reasoning-only", "Both"]

# Pastel palette sampled from the report's companion diagrams.
COLORS = {
    "Correct, no hallucination": "#6F91B8",
    "Correct, hallucination": "#B9D6A4",
    "Incorrect, no hallucination": "#EBC58F",
    "Incorrect, hallucination": "#D98178",
    "Vision-only": "#6F91B8",
    "Reasoning-only": "#EBC58F",
    "Both": "#D98178",
}
HEATMAP_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "report_warm",
    ["#FFF8ED", "#F0C59D", "#D98178", "#9F5147"],
)
JOINT_COUNTS = pd.DataFrame(
    {
        "Correct, no hallucination": [624, 1189, 1249],
        "Correct, hallucination": [466, 240, 226],
        "Incorrect, no hallucination": [86, 232, 272],
        "Incorrect, hallucination": [1034, 554, 495],
    },
    index=MODELS,
)

TYPE_COUNTS = pd.DataFrame(
    {
        "Vision-only": [267, 147, 158],
        "Reasoning-only": [183, 225, 169],
        "Both": [1050, 422, 394],
    },
    index=MODELS,
)


def validate_counts() -> None:
    """Cross-check the report counts and all relevant marginal totals."""
    assert JOINT_COUNTS.sum(axis=1).tolist() == [2210, 2215, 2242]
    assert JOINT_COUNTS.sum().tolist() == [3062, 932, 590, 2083]
    assert int(JOINT_COUNTS.to_numpy().sum()) == 6667

    hallucination_totals = (
        JOINT_COUNTS["Correct, hallucination"]
        + JOINT_COUNTS["Incorrect, hallucination"]
    )
    assert hallucination_totals.tolist() == [1500, 794, 721]
    assert TYPE_COUNTS.sum(axis=1).equals(hallucination_totals)
    assert TYPE_COUNTS.sum().tolist() == [572, 577, 1866]
    assert int(TYPE_COUNTS.to_numpy().sum()) == 3015


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 8.5,
            "axes.titlesize": 9.5,
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
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(
            FIG_DIR / f"{stem}.{extension}",
            dpi=600 if extension == "png" else None,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    with Image.open(FIG_DIR / f"{stem}.png") as image:
        image.convert("L").save(FIG_DIR / f"{stem}_grayscale.png", dpi=(600, 600))
    plt.close(fig)


def figure_label_joint() -> None:
    proportions = JOINT_COUNTS.div(JOINT_COUNTS.sum(axis=1), axis=0)
    correct = JOINT_COUNTS["Correct, no hallucination"] + JOINT_COUNTS["Correct, hallucination"]
    incorrect = JOINT_COUNTS["Incorrect, no hallucination"] + JOINT_COUNTS["Incorrect, hallucination"]
    conditional = pd.DataFrame(
        {
            "Correct\nanswer": JOINT_COUNTS["Correct, hallucination"] / correct,
            "Incorrect\nanswer": JOINT_COUNTS["Incorrect, hallucination"] / incorrect,
        },
        index=MODELS,
    )

    fig, (ax_bar, ax_heat) = plt.subplots(
        1,
        2,
        figsize=(6.7, 3.05),
        gridspec_kw={"width_ratios": [1.38, 1.0], "wspace": 0.42},
    )

    bottom = pd.Series(0.0, index=MODELS)
    for label in JOINT_LABELS:
        values = proportions[label]
        bars = ax_bar.bar(
            MODELS,
            values,
            bottom=bottom,
            width=0.64,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.65,
            label=label,
        )
        for bar, value, base in zip(bars, values, bottom):
            if value >= 0.075:
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{value:.1%}",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    color="#202833"
                    if label == "Incorrect, hallucination"
                    else ("white" if label == "Correct, no hallucination" else "#202833"),
                    fontweight="bold",
                )
        bottom = bottom + values

    ax_bar.set_title("(a) Correctness × hallucination", fontweight="bold", pad=7)
    ax_bar.set_xlabel("Model")
    ax_bar.set_ylabel("Sample share")
    ax_bar.set_ylim(0, 1)
    ax_bar.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax_bar.grid(axis="x", visible=False)
    ax_bar.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        frameon=False,
        columnspacing=0.9,
        handlelength=1.2,
    )
    sns.despine(ax=ax_bar)

    annotations = conditional.map(lambda value: f"{value:.1%}")
    heatmap = sns.heatmap(
        conditional,
        ax=ax_heat,
        annot=annotations,
        fmt="",
        cmap=HEATMAP_CMAP,
        vmin=0,
        vmax=1,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={
            "label": "Hallucination rate",
            "format": mpl.ticker.PercentFormatter(1.0, decimals=0),
            "shrink": 0.84,
            "pad": 0.04,
        },
        annot_kws={"fontsize": 8.5, "fontweight": "bold"},
    )
    colorbar = heatmap.collections[0].colorbar
    if colorbar is not None and colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    ax_heat.set_title("(b) Conditional hallucination rate", fontweight="bold", pad=7)
    ax_heat.set_xlabel("Condition")
    ax_heat.set_ylabel("")
    ax_heat.tick_params(axis="x", rotation=0)
    ax_heat.tick_params(axis="y", rotation=0)

    save_figure(fig, "fig_3_1_2a_label_joint")


def figure_hallucination_types() -> None:
    totals = TYPE_COUNTS.sum(axis=1)
    proportions = TYPE_COUNTS.div(totals, axis=0)
    colors = [COLORS[label] for label in TYPE_LABELS]

    fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.10), subplot_kw={"aspect": "equal"})
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.14, top=0.98, wspace=0.28)
    for ax, model in zip(axes, MODELS):
        values = proportions.loc[model, TYPE_LABELS].to_numpy()
        wedges, _ = ax.pie(
            values,
            startangle=90,
            counterclock=False,
            colors=colors,
            wedgeprops={"width": 0.26, "edgecolor": "white", "linewidth": 1.0},
        )
        for wedge, label, value in zip(wedges, TYPE_LABELS, values):
            angle = math.radians((wedge.theta1 + wedge.theta2) / 2)
            ax.text(
                0.87 * math.cos(angle),
                0.87 * math.sin(angle),
                f"{value:.1%}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="#000000",
                fontweight="bold",
            )
        ax.text(0, 0.06, model, ha="center", va="center", fontsize=8.2, fontweight="bold")
        ax.text(
            0,
            -0.10,
            f"n={int(totals.loc[model]):,}",
            ha="center",
            va="center",
            fontsize=6.6,
            color="#4B5563",
        )
        ax.set_axis_off()

    handles = [
        mpl.patches.Patch(
            facecolor=COLORS[label],
            edgecolor="white",
            label=label,
        )
        for label in TYPE_LABELS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3,
        frameon=False,
        columnspacing=1.3,
        handlelength=1.5,
    )

    save_figure(fig, "fig_3_1_2b_hallucination_types")


def main() -> None:
    validate_counts()
    configure_style()
    figure_label_joint()
    figure_hallucination_types()


if __name__ == "__main__":
    main()
