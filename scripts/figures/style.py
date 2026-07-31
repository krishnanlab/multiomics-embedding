"""
Author: Keenan Manpearl
Date: 2026-07-31

Shared matplotlib rcParams for journal-quality figures - high DPI, tight
margins, no chart-junk. Import and call apply() at the top of every
scripts/figures/*.py script before creating any figure, so every figure in
results/figures/ shares one visual style.
"""

import matplotlib.pyplot as plt

DPI = 600


def apply() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.size": 12,
            "font.family": "sans-serif",
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def savefig(fig, out_path) -> None:
    """Save at the shared DPI/bbox settings and close the figure - every
    scripts/figures/*.py script generates many figures in one run, so
    closing each one after saving keeps memory from growing unbounded."""
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
