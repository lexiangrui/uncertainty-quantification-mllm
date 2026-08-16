"""Generate the conceptual uncertainty-hallucination quadrant figure."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


SKILL_SCRIPTS = Path("/Users/lexiangrui/.codex/skills/scipilot-figure-skill/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))

from export_figure import export_figure  # noqa: E402
from setup_style import setup_style  # noqa: E402
from visual_qa import audit_layout, print_report, render_preview  # noqa: E402


OUT_BASENAME = Path(__file__).resolve().parents[1] / "figures" / "uncertainty_hallucination_quadrants"
PREVIEW = Path(__file__).resolve().parents[1] / "figures" / "_uncertainty_hallucination_quadrants_preview.png"


def add_quadrant(
    ax,
    xy,
    facecolor,
    title,
    descriptor,
    *,
    edgecolor="none",
    linewidth=0.0,
    title_color="#23313F",
    title_weight="semibold",
    badge=None,
):
    """Add one labeled quadrant without implying a numeric data scale."""
    patch = Rectangle(
        xy,
        1,
        1,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        joinstyle="miter",
        zorder=1,
    )
    ax.add_patch(patch)
    x, y = xy
    ax.text(
        x + 0.5,
        y + 0.59,
        title,
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight=title_weight,
        color=title_color,
        linespacing=1.12,
        zorder=5,
    )
    ax.text(
        x + 0.5,
        y + 0.27,
        descriptor,
        ha="center",
        va="center",
        fontsize=7.2,
        color="#46535E",
        linespacing=1.12,
        zorder=5,
    )
    if badge:
        ax.text(
            x + 0.90,
            y + 0.88,
            badge,
            ha="right",
            va="center",
            fontsize=7.1,
            fontweight="bold",
            color="white",
            bbox={
                "boxstyle": "round,pad=0.24,rounding_size=0.10",
                "facecolor": "#9E2F2F",
                "edgecolor": "none",
            },
            zorder=6,
        )


def main() -> None:
    # The report is an English LaTeX manuscript; use a serif style that blends
    # with the surrounding text while retaining publication-safe TrueType PDF
    # fonts through the skill's shared export helper.
    setup_style(journal="ieee", lang="en", use_sciplots=False, constrained_layout=False)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })

    fig, ax = plt.subplots(figsize=(6.50, 3.35))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Direct labels carry the categorical meaning. Low-saturation neutral
    # fills keep the three background cases subordinate, while LUH receives a
    # warm accent, thick border, and badge as redundant emphasis.
    add_quadrant(
        ax,
        (0, 0),
        "#E4EEF5",
        "Certain and reliable\nresponse",
        "stable and grounded",
    )
    add_quadrant(
        ax,
        (0, 1),
        "#F0F4F7",
        "High-uncertainty\nnon-hallucination",
        "ambiguous but grounded",
    )
    add_quadrant(
        ax,
        (1, 1),
        "#F3F0EC",
        "High-uncertainty\nhallucination",
        "unstable and unsupported",
    )
    add_quadrant(
        ax,
        (1, 0),
        "#F2B894",
        "Low-uncertainty\nhallucination",
        "stable but unsupported\nUQ false negative",
        edgecolor="#9E2F2F",
        linewidth=2.4,
        title_color="#7A2020",
        title_weight="bold",
        badge="LUH",
    )

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("auto")
    ax.set_xticks([0.5, 1.5], ["No hallucination", "Hallucination"])
    ax.set_yticks([0.5, 1.5], ["Low uncertainty", "High uncertainty"])
    ax.tick_params(axis="both", length=0, pad=7, labelsize=8.2)
    ax.set_xlabel("Hallucination status", fontsize=9.2, fontweight="semibold", labelpad=9)
    ax.set_ylabel("Uncertainty level", fontsize=9.2, fontweight="semibold", labelpad=9)

    # Keep the classification boundaries visually explicit but subordinate to
    # the quadrant fills and LUH border.
    ax.axvline(1, color="#67727C", linewidth=0.85, zorder=4)
    ax.axhline(1, color="#67727C", linewidth=0.85, zorder=4)
    ax.add_patch(
        Rectangle(
            (0, 0),
            2,
            2,
            facecolor="none",
            edgecolor="#67727C",
            linewidth=0.9,
            zorder=4,
        )
    )
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.24, top=0.97)

    # First render a preview and run deterministic layout checks before the
    # final vector/raster export, following the figure skill's QA workflow.
    render_preview(fig, str(PREVIEW), dpi=150)
    print_report(audit_layout(fig))
    export_figure(
        fig,
        str(OUT_BASENAME),
        formats=["pdf", "png"],
        size_inches=(6.50, 3.35),
        dpi=600,
        grayscale_preview=True,
        tight=True,
        pad_inches=0.06,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
