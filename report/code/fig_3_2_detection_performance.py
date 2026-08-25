"""Generate grouped AUROC bar charts for report sections 3.2.1 and 3.2.2."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "report" / "figures"

MODELS = ["LLaVA", "Qwen", "InternVL"]
DATASETS = ["ViLP", "HallusionBench", "MM-Vet"]
DATASET_LABELS = ["ViLP", "Hallusion-\nBench", "MM-Vet"]
METHODS = ["PPL", "SE", "UMPIRE"]
COLORS = {"PPL": "#6F91B8", "SE": "#B9D6A4", "UMPIRE": "#D98178"}

# Each tuple is (AUROC, CI low, CI high), in model/dataset/method order.
ERROR_RESULTS = {
    "LLaVA": {
        "ViLP": {"PPL": (0.575, 0.540, 0.611), "SE": (0.655, 0.622, 0.690), "UMPIRE": (0.609, 0.573, 0.645)},
        "HallusionBench": {"PPL": (0.567, 0.538, 0.595), "SE": (0.553, 0.519, 0.586), "UMPIRE": (0.579, 0.547, 0.612)},
        "MM-Vet": {"PPL": (0.758, 0.671, 0.834), "SE": (0.808, 0.727, 0.875), "UMPIRE": (0.786, 0.696, 0.861)},
    },
    "Qwen": {
        "ViLP": {"PPL": (0.612, 0.578, 0.649), "SE": (0.717, 0.680, 0.750), "UMPIRE": (0.653, 0.617, 0.687)},
        "HallusionBench": {"PPL": (0.643, 0.607, 0.677), "SE": (0.653, 0.615, 0.693), "UMPIRE": (0.629, 0.589, 0.668)},
        "MM-Vet": {"PPL": (0.717, 0.650, 0.785), "SE": (0.816, 0.759, 0.870), "UMPIRE": (0.769, 0.704, 0.831)},
    },
    "InternVL": {
        "ViLP": {"PPL": (0.622, 0.591, 0.657), "SE": (0.719, 0.686, 0.752), "UMPIRE": (0.640, 0.608, 0.675)},
        "HallusionBench": {"PPL": (0.708, 0.674, 0.741), "SE": (0.746, 0.713, 0.777), "UMPIRE": (0.674, 0.635, 0.712)},
        "MM-Vet": {"PPL": (0.724, 0.654, 0.791), "SE": (0.847, 0.790, 0.899), "UMPIRE": (0.828, 0.765, 0.886)},
    },
}

HALLUCINATION_RESULTS = {
    "LLaVA": {
        "ViLP": {"PPL": (0.541, 0.504, 0.580), "SE": (0.605, 0.569, 0.640), "UMPIRE": (0.570, 0.531, 0.609)},
        "HallusionBench": {"PPL": (0.607, 0.565, 0.646), "SE": (0.568, 0.518, 0.620), "UMPIRE": (0.588, 0.541, 0.633)},
        "MM-Vet": {"PPL": (0.567, 0.475, 0.651), "SE": (0.697, 0.618, 0.769), "UMPIRE": (0.679, 0.592, 0.758)},
    },
    "Qwen": {
        "ViLP": {"PPL": (0.501, 0.458, 0.544), "SE": (0.609, 0.565, 0.647), "UMPIRE": (0.546, 0.501, 0.594)},
        "HallusionBench": {"PPL": (0.558, 0.511, 0.602), "SE": (0.639, 0.602, 0.680), "UMPIRE": (0.613, 0.569, 0.658)},
        "MM-Vet": {"PPL": (0.536, 0.455, 0.621), "SE": (0.659, 0.567, 0.740), "UMPIRE": (0.576, 0.489, 0.664)},
    },
    "InternVL": {
        "ViLP": {"PPL": (0.533, 0.492, 0.575), "SE": (0.657, 0.617, 0.696), "UMPIRE": (0.563, 0.525, 0.603)},
        "HallusionBench": {"PPL": (0.640, 0.595, 0.685), "SE": (0.686, 0.647, 0.722), "UMPIRE": (0.661, 0.616, 0.701)},
        "MM-Vet": {"PPL": (0.632, 0.544, 0.710), "SE": (0.779, 0.706, 0.845), "UMPIRE": (0.720, 0.645, 0.779)},
    },
}


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
            "xtick.labelsize": 7.2,
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


def validate_results() -> None:
    expected_means = {
        "error": {"PPL": 0.658, "SE": 0.724, "UMPIRE": 0.685},
        "hallucination": {"PPL": 0.568, "SE": 0.655, "UMPIRE": 0.613},
    }
    for name, results in (("error", ERROR_RESULTS), ("hallucination", HALLUCINATION_RESULTS)):
        for method in METHODS:
            values = [results[model][dataset][method][0] for model in MODELS for dataset in DATASETS]
            assert round(float(np.mean(values)), 3) == expected_means[name][method]
        for model in MODELS:
            for dataset in DATASETS:
                for method in METHODS:
                    value, low, high = results[model][dataset][method]
                    assert 0 <= low <= value <= high <= 1


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


def plot_detection(results: dict, stem: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.7, 2.55), sharey=True)
    x = np.arange(len(DATASETS))
    width = 0.24

    for ax, model in zip(axes, MODELS):
        for method_index, method in enumerate(METHODS):
            values = np.array([results[model][dataset][method][0] for dataset in DATASETS])
            lows = np.array([results[model][dataset][method][1] for dataset in DATASETS])
            highs = np.array([results[model][dataset][method][2] for dataset in DATASETS])
            offset = (method_index - 1) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                color=COLORS[method],
                edgecolor="white",
                linewidth=0.7,
                label=method,
                zorder=2,
            )
            ax.errorbar(
                x + offset,
                values,
                yerr=np.vstack((values - lows, highs - values)),
                fmt="none",
                ecolor="#202833",
                elinewidth=0.75,
                capsize=2.0,
                capthick=0.75,
                zorder=3,
            )

        ax.axhline(0.5, color="#6B7280", linewidth=0.8, linestyle="--", zorder=1)
        ax.set_title(model, fontweight="bold", pad=6)
        ax.set_xticks(x, DATASET_LABELS)
        ax.set_xlabel("Dataset")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax)

    axes[0].set_ylabel("AUROC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handlelength=1.5,
    )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.27, top=0.92, wspace=0.14)
    save_figure(fig, stem)


def main() -> None:
    validate_results()
    configure_style()
    plot_detection(ERROR_RESULTS, "fig_3_2_1_error_detection")
    plot_detection(HALLUCINATION_RESULTS, "fig_3_2_2_hallucination_detection")


if __name__ == "__main__":
    main()
