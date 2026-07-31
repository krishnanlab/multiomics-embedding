"""
Author: Keenan Manpearl
Date: 2026-07-31

Effect-size and %-missingness figures: for each of the three methods compared
against Differential Abundance (DA) for `time` (Permutation Testing, Majority
Vote, Unanimous Vote), splits every feature into three detection-outcome
segments - DA only, both, method only - and boxplots two per-feature
covariates (DA's |Fold Change|, and % of samples missing a measurement) per
segment, with Mann-Whitney U significance brackets between segments. Adapted
from the segment-boxplot cells in notebooks/2026-07-30_condensed_summary.ipynb,
generalized from permutation-testing-only to all three methods.

Also writes one unsegmented reference figure, `pct_missing_all_features.png`
(microbiome vs. metabolites, no DA/method segmentation), so the per-method
segments can be read against each omics type's overall missingness.

Run scripts/compute_majority_hits.py first to produce num_hits.tsv.
Writes 7 PNGs to results/figures.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from scripts import da_comparison as dac
from scripts.figures import style
from scripts.figures.confusion_matrices import (
    LABEL_CONFIG,
    MAJORITY_MIN_HITS,
    N_EMBEDDINGS,
    PVAL_CUTOFF,
    TITLE_MAJORITY,
    TITLE_PERMUTATION,
    TITLE_UNANIMOUS,
    UNANIMOUS_MIN_HITS,
    load_da,
    load_majority_hits,
    load_omics_ids,
    load_permutation,
    vote_ids,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

LABEL = "time"  # diet DA is degenerate (zero significant features) - see confusion_matrices.py
OMICS_NAMES = ["microbiome", "metabolites"]

COLOR_ONLY_BASELINE = "#0072B2"
COLOR_BOTH = "#009E73"
COLOR_ONLY_CONSENSUS = "#E69F00"
SEGMENT_ORDER = ["only_baseline", "both", "only_consensus"]
SEGMENT_COLORS = {"only_baseline": COLOR_ONLY_BASELINE, "both": COLOR_BOTH, "only_consensus": COLOR_ONLY_CONSENSUS}
SIG_PAIRS = [(0, 1), (1, 2), (0, 2)]  # (DA-only,both), (both,method-only), (DA-only,method-only)


def _p_to_stars(p: float) -> str:
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def segment_tick_labels(method_title: str) -> dict:
    return {
        "only_baseline": "differential\nabundance only",
        "both": "both\nmethods",
        "only_consensus": f"{method_title.lower()}\nonly",
    }


def boxplot_by_segment(
    ax,
    values: pd.Series,
    seg: dict,
    tick_labels: dict,
    title: str,
    ylabel: "str | None",
) -> "float | None":
    """Boxplot a per-feature Series split into the three comparison segments
    (only_baseline/both/only_consensus, fixed color/order), annotated with
    pairwise Mann-Whitney U significance brackets between every pair of
    segments. Returns the y value needed to keep the topmost bracket in view
    (None if there was no data to plot) instead of calling ax.set_ylim
    itself, so a caller sharing the y-axis across subplots can take the max
    over all of them first."""
    data = [values.reindex(list(seg[s])).dropna() for s in SEGMENT_ORDER]
    bp = ax.boxplot(
        data,
        tick_labels=[tick_labels[s] for s in SEGMENT_ORDER],
        showfliers=False,
        patch_artist=True,
        widths=0.5,
    )
    for patch, s in zip(bp["boxes"], SEGMENT_ORDER):
        patch.set_facecolor(SEGMENT_COLORS[s])
        patch.set_alpha(0.7)
    ax.set_title(title, fontsize=11.5)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    nonempty = [d for d in data if len(d) >= 2]
    if not nonempty:
        return None
    all_vals = pd.concat(nonempty)
    data_range = float(all_vals.max() - all_vals.min()) or 1.0
    cap_ys = [cap.get_ydata()[0] for i, cap in enumerate(bp["caps"]) if len(data[i // 2]) > 0]
    visible_top = max(cap_ys) if cap_ys else float(all_vals.max())
    step = data_range * 0.08
    for level, (i, j) in enumerate(SIG_PAIRS):
        a, b = data[i], data[j]
        if len(a) < 2 or len(b) < 2:
            continue
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        y = visible_top + step * (level + 0.6)
        ax.plot([i + 1, i + 1, j + 1, j + 1], [y, y + step * 0.25, y + step * 0.25, y], color="0.3", linewidth=0.8)
        ax.text((i + j) / 2 + 1, y + step * 0.3, _p_to_stars(p), ha="center", va="bottom", fontsize=11)
    return visible_top + step * (len(SIG_PAIRS) + 1.2)


def consensus_ids_by_method(cons_by_omics: dict, majority_hits: pd.DataFrame, omics_ids: dict) -> dict:
    """{method_title: {omics: set of feature ids the method calls associated,
    direction-agnostic - i.e. associated with *either* direction}}."""
    directions = LABEL_CONFIG[LABEL]["da_directions"]

    def permutation_consensus(omics: str) -> set:
        return dac.select_features(cons_by_omics[omics], "p_value", "<", PVAL_CUTOFF)

    def vote_consensus(omics: str, min_hits: int) -> set:
        result: set = set()
        for direction in directions:
            result |= vote_ids(majority_hits, omics_ids, LABEL, direction, omics, min_hits)
        return result

    return {
        TITLE_PERMUTATION: {omics: permutation_consensus(omics) for omics in OMICS_NAMES},
        TITLE_MAJORITY: {omics: vote_consensus(omics, MAJORITY_MIN_HITS) for omics in OMICS_NAMES},
        TITLE_UNANIMOUS: {omics: vote_consensus(omics, UNANIMOUS_MIN_HITS) for omics in OMICS_NAMES},
    }


def method_suffix(method_title: str) -> str:
    if method_title == TITLE_PERMUTATION:
        return f"permutation testing significant: raw p<{PVAL_CUTOFF}"
    min_hits = MAJORITY_MIN_HITS if method_title == TITLE_MAJORITY else UNANIMOUS_MIN_HITS
    return f"{method_title.lower()} significant: >={min_hits}/{N_EMBEDDINGS} deployment embeddings with prob>0.5"


def plot_covariate(
    covariate: dict,
    ylabel: str,
    method_title: str,
    segments: dict,
    suptitle: str,
    out_path: Path,
    sharey: bool = False,
) -> None:
    tick_labels = segment_tick_labels(method_title)
    fig, axes = plt.subplots(
        1,
        len(OMICS_NAMES),
        figsize=(5.8 * len(OMICS_NAMES), 5.2),
        squeeze=False,
        sharey=sharey,
        layout="constrained",
    )
    axes = axes[0]
    tops = []
    for j, omics in enumerate(OMICS_NAMES):
        seg = segments[omics]
        title = (
            f"{omics.capitalize()}\n(n: DA only={seg['n_only_baseline']}, both={seg['n_both']}, "
            f"{method_title.lower()} only={seg['n_only_consensus']})"
        )
        top = boxplot_by_segment(
            axes[j],
            covariate[omics],
            seg,
            tick_labels,
            title=title,
            ylabel=ylabel if j == 0 else None,
        )
        if top is not None:
            if sharey:
                tops.append(top)
            else:
                axes[j].set_ylim(top=top)
        if sharey and j > 0:
            axes[j].tick_params(axis="y", labelleft=False)
    if sharey and tops:
        axes[0].set_ylim(top=max(tops))
    fig.suptitle(suptitle, fontsize=15)
    style.savefig(fig, out_path)
    print(f"wrote {out_path}")


def plot_pct_missing_all(pct_missing: dict, out_path: Path) -> None:
    """Single-panel reference figure: % samples missing per feature across
    every feature of each omics type (no DA/method segmentation), so the
    per-method segment plots can be read against each omics type's overall
    missingness. Adapted from the microbes-vs-metabolites boxplot cell in
    notebooks/2026-07-30_condensed_summary.ipynb."""
    data = [pct_missing[omics] for omics in OMICS_NAMES]
    colors = [COLOR_ONLY_BASELINE, COLOR_ONLY_CONSENSUS]
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    bp = ax.boxplot(
        data,
        tick_labels=[omics.capitalize() for omics in OMICS_NAMES],
        showfliers=False,
        patch_artist=True,
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("% samples missing")
    ax.spines[["top", "right"]].set_visible(False)

    all_vals = pd.concat(data)
    data_range = float(all_vals.max() - all_vals.min()) or 1.0
    step = data_range * 0.08
    cap_ys = [cap.get_ydata()[0] for cap in bp["caps"]]
    visible_top = max(cap_ys) if cap_ys else float(all_vals.max())
    _, p = stats.mannwhitneyu(data[0], data[1], alternative="two-sided")
    y = visible_top + step * 0.6
    ax.plot([1, 1, 2, 2], [y, y + step * 0.25, y + step * 0.25, y], color="0.3", linewidth=0.8)
    ax.text(1.5, y + step * 0.3, _p_to_stars(p), ha="center", va="bottom", fontsize=11)
    ax.set_ylim(top=y + step * 1.2)
    ax.set_title(
        "Percent of samples missing per feature, all features\n"
        f"(n: microbiome={len(data[0])}, metabolites={len(data[1])})",
        fontsize=15,
    )
    style.savefig(fig, out_path)
    print(f"wrote {out_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    style.apply()
    omics_ids = load_omics_ids(REPO_ROOT)

    da_by_omics = {omics: load_da(REPO_ROOT, LABEL, omics) for omics in OMICS_NAMES}
    baseline_ids = {omics: dac.select_features(da_by_omics[omics], "FDR", "<", PVAL_CUTOFF) for omics in OMICS_NAMES}

    cons_by_omics = load_permutation(REPO_ROOT, LABEL)
    majority_hits = load_majority_hits(REPO_ROOT)
    consensus_by_method = consensus_ids_by_method(cons_by_omics, majority_hits, omics_ids)

    segments_by_method = {
        method_title: {
            omics: dac.compare_feature_sets(
                baseline_ids[omics], consensus_by_method[method_title][omics], "baseline", "consensus"
            )
            for omics in OMICS_NAMES
        }
        for method_title in consensus_by_method
    }

    effect_size = {omics: da_by_omics[omics]["Fold Change"].abs() for omics in OMICS_NAMES}

    print("computing per-feature missingness...")
    edges = pd.read_csv(REPO_ROOT / "raw_data" / "edges.tsv", sep="\t", header=None, names=["sample", "feature", "weight"])
    total_samples = edges["sample"].nunique()
    degree = dac.per_feature_degree(str(REPO_ROOT / "raw_data" / "edges.tsv"))
    pct_missing_full = 100 * (1 - degree / total_samples)
    pct_missing = {omics: pct_missing_full.reindex(sorted(omics_ids[omics])).dropna() for omics in OMICS_NAMES}

    for method_title in [TITLE_PERMUTATION, TITLE_MAJORITY, TITLE_UNANIMOUS]:
        slug = method_title.lower().replace(" ", "_")
        segments = segments_by_method[method_title]
        suffix = method_suffix(method_title)

        plot_covariate(
            effect_size,
            "|Fold Change| (log2)",
            method_title,
            segments,
            f"Differential-abundance effect size ({LABEL})\n"
            f"(DA significant: FDR<{PVAL_CUTOFF}; {suffix})",
            out_dir / f"effect_size_da_vs_{slug}_{LABEL}.png",
        )
        plot_covariate(
            pct_missing,
            "% samples missing",
            method_title,
            segments,
            f"Percent of samples missing identified feature ({LABEL})\n"
            f"(DA significant: FDR<{PVAL_CUTOFF}; {suffix})",
            out_dir / f"pct_missing_da_vs_{slug}_{LABEL}.png",
            sharey=True,
        )

    plot_pct_missing_all(pct_missing, out_dir / "pct_missing_all_features.png")


if __name__ == "__main__":
    main()
