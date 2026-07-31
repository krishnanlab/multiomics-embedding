"""
Author: Keenan Manpearl
Date: 2026-07-31

TSV + heatmap summaries of how three feature-detection methods (permutation
testing's consensus modes, and majority/unanimous vote) agree with baseline
differential abundance (DA) for the time label, generalizing the mode x
category comparison in
notebooks/2026-07-28_time_consensus_mode_comparison.ipynb into a script.
Diet DA found zero significant features, so it's excluded (same as
scripts/figures/confusion_matrices.py).

Writes two directories under --out-dir, each with one TSV+PNG per metric in
METRICS:

- permutation_modes/: one row per consensus mode, auto-discovered from
  results/permutations/time/100000_permutations/combined_<mode>.tsv.
- consensus_modes/: 3 rows - RECOMMENDED_MODE, majority_vote, and
  unanimous_vote (results/deployment_for_permutations/num_hits.tsv,
  thresholded via scripts/compute_majority_hits.py).

See results/figures/README.md for what each category column means and
results/figures/summaries/README.md for what each metric means.
direction_agreement is only computed for "(combined)" categories (NaN
elsewhere - see compute_row).
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize

from scripts import da_comparison as dac
from scripts.figures import style

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PVAL_CUTOFF = 0.05
MAJORITY_MIN_HITS = 4
UNANIMOUS_MIN_HITS = 7

DIRECTION_MAP = {"baseline": "5-month-old", "endpoint": "12-month-old"}
GROUP_TO_DIRECTION = {v: k for k, v in DIRECTION_MAP.items()}
DIRECTIONS = ["5-month-old", "12-month-old"]

METRICS = [
    "n_features_identified",
    "jaccard",
    "overlap_coefficient",
    "direction_agreement",
    "accuracy",
    "precision",
    "recall",
]


def category_label(omics: str, direction: str) -> str:
    return f"{omics} ({direction})" if direction == "combined" else f"{omics}-{direction}"


def build_categories(micro_da, metab_da, combined_da) -> list:
    """(omics_name, direction_or_"combined", da_df) - matches the reference
    notebook's CATEGORIES: per-omics-per-direction, per-omics-combined
    (both directions pooled), and "everything (combined)" (both omics types
    and both directions pooled)."""
    return [
        ("microbiome", "5-month-old", micro_da),
        ("microbiome", "12-month-old", micro_da),
        ("microbiome", "combined", micro_da),
        ("metabolites", "5-month-old", metab_da),
        ("metabolites", "12-month-old", metab_da),
        ("metabolites", "combined", metab_da),
        ("everything", "combined", combined_da),
    ]


def confusion_metrics(baseline_ids: set, consensus_ids: set, universe: set) -> dict:
    tp = len(baseline_ids & consensus_ids)
    fn = len(baseline_ids - consensus_ids)
    fp = len(consensus_ids - baseline_ids)
    tn = len(universe) - tp - fn - fp
    n = tp + fn + fp + tn
    return {
        "accuracy": (tp + tn) / n if n else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
    }


def compute_row(da_df, baseline_ids: set, consensus_ids: set, direction_df, universe: set) -> dict:
    """One row's worth of every metric. direction_df: a DataFrame with a
    "direction" column (real consensus mode table, or a synthetic one built
    from vote thresholds - see direction_frame_for_votes) used only to
    compute direction_agreement over the baseline&consensus overlap; pass
    None to force NaN (non-"combined" categories - see module docstring)."""
    overlap = dac.compare_feature_sets(baseline_ids, consensus_ids, "baseline", "consensus")
    if direction_df is not None:
        _, direction_agreement = dac.direction_agreement(da_df, direction_df, overlap["both"], DIRECTION_MAP)
    else:
        direction_agreement = float("nan")
    row = {
        "n_features_identified": len(consensus_ids),
        "jaccard": overlap["jaccard"],
        "overlap_coefficient": overlap["overlap_coefficient"],
        "direction_agreement": direction_agreement,
    }
    row.update(confusion_metrics(baseline_ids, consensus_ids, universe))
    return row


def direction_frame_for_votes(hits_by_direction: dict) -> pd.DataFrame:
    """Build a synthetic {"direction": ...} DataFrame from {direction_label:
    feature_id_set} (as returned by vote-thresholding each direction
    separately), for dac.direction_agreement to consume the same way it
    consumes a real consensus mode table's "direction" column. A feature can
    only clear a >=4/7 (or >=7/7) vote threshold in one direction at a time
    (dairy/meat, or baseline/endpoint, hit-counts sum to 7 - two directions
    both >=4 would need >=8), so this mapping is always well-defined."""
    assignment = {}
    for direction_label, ids in hits_by_direction.items():
        internal = GROUP_TO_DIRECTION[direction_label]
        for feature_id in ids:
            assignment[feature_id] = internal
    return pd.DataFrame({"direction": pd.Series(assignment)})


def permutation_testing_rows(micro_da, metab_da, combined_da, results_dir: Path) -> pd.DataFrame:
    modes = dac.discover_consensus_modes(str(results_dir))
    mode_tables = dac.load_consensus_modes(str(results_dir), modes=modes)
    micro_tables = {m: dac.restrict_to_group(df, str(REPO_ROOT / "data/nodes/microbes.txt")) for m, df in mode_tables.items()}
    metab_tables = {m: dac.restrict_to_group(df, str(REPO_ROOT / "data/nodes/metabolites.txt")) for m, df in mode_tables.items()}
    tables_by_omics = {"microbiome": micro_tables, "metabolites": metab_tables, "everything": mode_tables}
    categories = build_categories(micro_da, metab_da, combined_da)

    rows = []
    for mode in modes:
        for omics_name, direction, da_df in categories:
            cons_df = tables_by_omics[omics_name][mode]
            baseline_ids = dac.select_features(da_df, "FDR", "<", PVAL_CUTOFF)
            consensus_ids = dac.select_features(cons_df, "p_value", "<", PVAL_CUTOFF)
            if direction != "combined":
                baseline_ids &= dac.select_features(da_df, dac.BASELINE_GROUP_COL, "==", direction)
                consensus_ids &= dac.select_features(cons_df, "direction", "==", GROUP_TO_DIRECTION[direction])
            direction_df = cons_df if direction == "combined" else None
            row = compute_row(da_df, baseline_ids, consensus_ids, direction_df, set(da_df.index))
            row.update({"mode": mode, "category": category_label(omics_name, direction)})
            rows.append(row)
    return pd.DataFrame(rows)


def vote_rows(micro_da, metab_da, combined_da, omics_ids: dict, num_hits: pd.DataFrame, method_name: str, min_hits: int) -> pd.DataFrame:
    categories = build_categories(micro_da, metab_da, combined_da)
    hit_cols = {d: f"time_{GROUP_TO_DIRECTION[d]}" for d in DIRECTIONS}
    all_ids = {"microbiome": omics_ids["microbiome"], "metabolites": omics_ids["metabolites"], "everything": omics_ids["microbiome"] | omics_ids["metabolites"]}

    def vote_ids(direction: str, omics_name: str) -> set:
        hits = num_hits[hit_cols[direction]]
        return set(hits.index[hits >= min_hits]) & all_ids[omics_name]

    rows = []
    for omics_name, direction, da_df in categories:
        baseline_ids = dac.select_features(da_df, "FDR", "<", PVAL_CUTOFF)
        hits_by_direction = {d: vote_ids(d, omics_name) for d in DIRECTIONS}
        if direction != "combined":
            baseline_ids &= dac.select_features(da_df, dac.BASELINE_GROUP_COL, "==", direction)
            consensus_ids = hits_by_direction[direction]
            direction_df = None
        else:
            consensus_ids = hits_by_direction[DIRECTIONS[0]] | hits_by_direction[DIRECTIONS[1]]
            direction_df = direction_frame_for_votes(hits_by_direction)
        row = compute_row(da_df, baseline_ids, consensus_ids, direction_df, set(da_df.index))
        row.update({"mode": method_name, "category": category_label(omics_name, direction)})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric_heatmap(pivot: pd.DataFrame, metric: str, out_path: Path) -> None:
    """Heatmap version of a metric's (mode-or-method) x category TSV - same
    cell layout, annotated with each value ("n/a" for NaN, e.g. a
    one-directional mode's missing direction). Color limits are data-derived
    (this table's own min/max, ignoring NaN) rather than each metric's
    theoretical range, so the color scale always uses its full contrast on
    whatever this particular table actually contains. n_features_identified
    spans a huge range (single digits to 25,900 - even within one table) so
    it gets a log color scale (see scripts/figures/confusion_matrices.py's
    LogNorm use for the same reason); every other metric gets a linear scale."""
    values = pivot.to_numpy(dtype=float)
    finite = values[~np.isnan(values)]
    if metric == "n_features_identified":
        positive = finite[finite > 0]
        vmin = positive.min() if len(positive) else 1
        vmax = max(finite.max() if len(finite) else 1, vmin)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin = finite.min() if len(finite) else 0.0
        vmax = finite.max() if len(finite) else 1.0
        if vmin == vmax:
            vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(1.3 * len(pivot.columns) + 2.5, 0.55 * len(pivot.index) + 1.5))
    im = ax.imshow(values, cmap="Blues", norm=norm, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if np.isnan(val):
                ax.text(j, i, "n/a", ha="center", va="center", color="0.6", fontsize=8)
                continue
            text = f"{val:,.0f}" if metric == "n_features_identified" else f"{val:.2f}"
            color = "white" if norm(val) > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)
    ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index, fontsize=8)
    ax.set_title(metric.replace("_", " "))
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    style.savefig(fig, out_path)
    print(f"wrote {out_path}")


def write_metric_tsvs(df: pd.DataFrame, category_order: list, out_dir: Path) -> None:
    """direction_agreement is only ever computed for "(combined)" categories
    (see compute_row's docstring - it's NaN everywhere else by construction),
    so its table/heatmap drops the direction-specific columns entirely
    instead of showing them as all-NaN/"n/a"."""
    out_dir.mkdir(parents=True, exist_ok=True)
    style.apply()
    for metric in METRICS:
        columns = [c for c in category_order if "(combined)" in c] if metric == "direction_agreement" else category_order
        pivot = df.pivot(index="mode", columns="category", values=metric)[columns]
        pivot.index.name = "mode"
        out_path = out_dir / f"{metric}.tsv"
        pivot.to_csv(out_path, sep="\t")
        print(f"wrote {out_path}")
        plot_metric_heatmap(pivot, metric, out_dir / f"{metric}.png")


RECOMMENDED_MODE = "mean_confidence"


def main():
    parser = ArgumentParser()
    parser.add_argument("--out-dir", default="results/figures/summaries")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    micro_da = dac.load_baseline_da(str(REPO_ROOT / "results/differential_abundance/time_microbiome.txt"))
    metab_da = dac.load_baseline_da(str(REPO_ROOT / "results/differential_abundance/time_metabolites.txt"))
    combined_da = pd.concat([micro_da, metab_da])
    category_order = [category_label(o, d) for o, d, _ in build_categories(micro_da, metab_da, combined_da)]

    permutation_df = permutation_testing_rows(
        micro_da, metab_da, combined_da, REPO_ROOT / "results/permutations/time/100000_permutations"
    )
    write_metric_tsvs(permutation_df, category_order, out_dir / "permutation_modes")

    omics_ids = {}
    for omics, fname in [("microbiome", "microbes.txt"), ("metabolites", "metabolites.txt")]:
        with open(REPO_ROOT / "data/nodes" / fname) as f:
            omics_ids[omics] = {line.strip() for line in f if line.strip()}
    num_hits_path = REPO_ROOT / "results/deployment_for_permutations/num_hits.tsv"
    if not num_hits_path.exists():
        raise FileNotFoundError(f"{num_hits_path} not found - run scripts/compute_majority_hits.py first")
    num_hits = pd.read_csv(num_hits_path, sep="\t", index_col=0)

    vote_dfs = {
        method_name: vote_rows(micro_da, metab_da, combined_da, omics_ids, num_hits, method_name, min_hits)
        for method_name, min_hits in [("majority_vote", MAJORITY_MIN_HITS), ("unanimous_vote", UNANIMOUS_MIN_HITS)]
    }

    # headline comparison: the one permutation-testing mode used everywhere
    # else in results/figures (scripts/figures/confusion_matrices.py, and
    # the notebook's own recommendation - see RECOMMENDED_MODE) side by side
    # with majority vote and unanimous vote, instead of all 9 modes at once.
    recommended_df = permutation_df[permutation_df["mode"] == RECOMMENDED_MODE]
    consensus_modes_df = pd.concat([recommended_df, vote_dfs["majority_vote"], vote_dfs["unanimous_vote"]], ignore_index=True)
    write_metric_tsvs(consensus_modes_df, category_order, out_dir / "consensus_modes")


if __name__ == "__main__":
    main()
