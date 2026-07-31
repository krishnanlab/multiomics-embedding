"""
Author: Keenan Manpearl
Date: 2026-07-31

Confusion-matrix figures comparing four ways of calling a feature "associated"
for the diet/time labels: Differential Abundance (DA), Permutation Testing,
Majority Vote, and Unanimous Vote. Adapted from the confusion-matrix cells in
notebooks/2026-07-30_condensed_summary.ipynb.

The x-axis (columns) always follows a fixed preference order - DA, then
Permutation Testing, then Majority Vote, then Unanimous Vote - whichever of
the two methods being compared comes later in that order is on the x-axis.

Diet DA found zero significant features in either omics type, so DA
comparisons are time-only. Run scripts/compute_majority_hits.py first to
produce num_hits.tsv. Writes 9 PNGs to results/figures.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scripts import da_comparison as dac
from scripts.figures import style

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PVAL_CUTOFF = 0.05
N_EMBEDDINGS = 7
MAJORITY_MIN_HITS = 4
UNANIMOUS_MIN_HITS = 7

OMICS_NAMES = ["microbiome", "metabolites"]
OMICS_NODE_FILES = {"microbiome": "microbes.txt", "metabolites": "metabolites.txt"}

LABEL_CONFIG = {
    "diet": {
        "da_directions": ["Dairy", "Meat"],
        "group_to_column": {"Dairy": "dairy", "Meat": "meat"},
    },
    "time": {
        "da_directions": ["5-month-old", "12-month-old"],
        "group_to_column": {"5-month-old": "baseline", "12-month-old": "endpoint"},
    },
}

# diet DA (results/differential_abundance/diet_*.txt) found zero significant
# features in either omics type - any DA comparison for diet would be
# degenerate, so DA comparisons are only generated for these labels.
DA_LABELS = {"time"}

TITLE_DA = "Differential Abundance"
TITLE_PERMUTATION = "Permutation Testing"
TITLE_MAJORITY = "Majority Vote"
TITLE_UNANIMOUS = "Unanimous Vote"

SIG_LABELS = ["Associated", "Not Associated"]

# x-axis preference order, later wins - see module docstring
METHOD_ORDER = [TITLE_DA, TITLE_PERMUTATION, TITLE_MAJORITY, TITLE_UNANIMOUS]


def load_omics_ids(repo_root: Path) -> dict:
    ids = {}
    for omics, fname in OMICS_NODE_FILES.items():
        with open(repo_root / "data" / "nodes" / fname) as f:
            ids[omics] = {line.strip() for line in f if line.strip()}
    return ids


def load_da(repo_root: Path, label: str, omics: str):
    return dac.load_baseline_da(str(repo_root / "results" / "differential_abundance" / f"{label}_{omics}.txt"))


def load_permutation(repo_root: Path, label: str) -> dict:
    """{omics: mean_confidence combined table restricted to that omics}."""
    full = dac.load_consensus_modes(
        str(repo_root / "results" / "permutations" / label / "100000_permutations"), modes=["mean_confidence"]
    )["mean_confidence"]
    return {
        omics: dac.restrict_to_group(full, str(repo_root / "data" / "nodes" / fname))
        for omics, fname in OMICS_NODE_FILES.items()
    }


def load_majority_hits(repo_root: Path):
    path = repo_root / "results" / "deployment_for_permutations" / "num_hits.tsv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/compute_majority_hits.py first")
    return dac.pd.read_csv(path, sep="\t", index_col=0)


def confusion_counts(row_ids: set, col_ids: set, universe: set):
    tp = len(row_ids & col_ids)
    fn = len(row_ids - col_ids)
    fp = len(col_ids - row_ids)
    tn = len(universe) - tp - fn - fp
    return tp, fn, fp, tn


def plot_confusion_grid(omics_names, directions, cell_fn, row_title, col_title, suptitle, out_path):
    fig, axes = plt.subplots(
        len(directions),
        len(omics_names),
        figsize=(5.2 * len(omics_names), 5.0 * len(directions)),
        squeeze=False,
        layout="constrained",
    )
    for i, direction in enumerate(directions):
        for j, omics_name in enumerate(omics_names):
            ax = axes[i][j]
            tp, fn, fp, tn = cell_fn(omics_name, direction)
            mat = [[tp, fn], [fp, tn]]
            norm = LogNorm(vmin=1, vmax=max(tp, fn, fp, tn, 1))
            ax.imshow(mat, cmap="Blues", norm=norm)
            for r in range(2):
                for c in range(2):
                    val = mat[r][c]
                    color = "white" if norm(val) > 0.6 else "black"
                    ax.text(c, r, f"{val:,}", ha="center", va="center", color=color, fontsize=19)
            ax.set_xticks([0, 1], labels=SIG_LABELS)
            ax.set_yticks([0, 1], labels=SIG_LABELS)
            ax.set_xlabel(col_title)
            ax.set_ylabel(row_title)
            ax.set_title(f"{omics_name} - {direction}")
    fig.suptitle(suptitle)
    style.savefig(fig, out_path)
    print(f"wrote {out_path}")


def plot_confusion_grid_ordered(
    omics_names, directions, ids_fn_a, name_a, ids_fn_b, name_b, universe_fn, suptitle_fn, out_path
):
    """Like plot_confusion_grid, but picks the x-axis (columns) via the fixed
    METHOD_ORDER preference instead of taking row/col titles directly."""
    if METHOD_ORDER.index(name_a) <= METHOD_ORDER.index(name_b):
        row_ids_fn, row_title, col_ids_fn, col_title = ids_fn_a, name_a, ids_fn_b, name_b
    else:
        row_ids_fn, row_title, col_ids_fn, col_title = ids_fn_b, name_b, ids_fn_a, name_a

    def cell_fn(omics, direction):
        return confusion_counts(row_ids_fn(omics, direction), col_ids_fn(omics, direction), universe_fn(omics))

    plot_confusion_grid(omics_names, directions, cell_fn, row_title, col_title, suptitle_fn(row_title, col_title), out_path)


def da_ids(da_df, direction: str) -> set:
    return dac.select_features(da_df, "FDR", "<", PVAL_CUTOFF) & dac.select_features(
        da_df, dac.BASELINE_GROUP_COL, "==", direction
    )


def permutation_ids(cons_df, label: str, direction: str) -> set:
    column = LABEL_CONFIG[label]["group_to_column"][direction]
    return dac.select_features(cons_df, "p_value", "<", PVAL_CUTOFF) & dac.select_features(
        cons_df, "direction", "==", column
    )


def vote_ids(majority_hits, omics_ids, label: str, direction: str, omics: str, min_hits: int) -> set:
    column = LABEL_CONFIG[label]["group_to_column"][direction]
    hits = majority_hits[f"{label}_{column}"]
    return set(hits.index[hits >= min_hits]) & omics_ids[omics]


def main():
    parser = ArgumentParser()
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    style.apply()
    omics_ids = load_omics_ids(REPO_ROOT)
    majority_hits = load_majority_hits(REPO_ROOT)

    for label, cfg in LABEL_CONFIG.items():
        directions = cfg["da_directions"]
        has_da = label in DA_LABELS
        da_by_omics = {omics: load_da(REPO_ROOT, label, omics) for omics in OMICS_NAMES} if has_da else None
        cons_by_omics = load_permutation(REPO_ROOT, label)

        da_ids_fn = (lambda omics, direction: da_ids(da_by_omics[omics], direction)) if has_da else None
        da_universe_fn = (lambda omics: set(da_by_omics[omics].index)) if has_da else None
        permutation_ids_fn = lambda omics, direction: permutation_ids(cons_by_omics[omics], label, direction)
        permutation_universe_fn = lambda omics: set(cons_by_omics[omics].index)

        if has_da:
            plot_confusion_grid_ordered(
                OMICS_NAMES,
                directions,
                da_ids_fn,
                TITLE_DA,
                permutation_ids_fn,
                TITLE_PERMUTATION,
                da_universe_fn,
                lambda row_title, col_title: (
                    f"Agreement between {row_title} and {col_title} ({label})\n"
                    f"(DA significant: FDR<{PVAL_CUTOFF}; permutation testing significant: raw p<{PVAL_CUTOFF})"
                ),
                out_dir / f"cm_da_vs_permutation_testing_{label}.png",
            )

        for vote_title, min_hits in [(TITLE_MAJORITY, MAJORITY_MIN_HITS), (TITLE_UNANIMOUS, UNANIMOUS_MIN_HITS)]:
            vote_slug = vote_title.lower().replace(" ", "_")
            vote_ids_fn = lambda omics, direction, min_hits=min_hits: vote_ids(
                majority_hits, omics_ids, label, direction, omics, min_hits
            )

            if has_da:
                plot_confusion_grid_ordered(
                    OMICS_NAMES,
                    directions,
                    da_ids_fn,
                    TITLE_DA,
                    vote_ids_fn,
                    vote_title,
                    da_universe_fn,
                    lambda row_title, col_title, vote_title=vote_title, min_hits=min_hits: (
                        f"Agreement between {row_title} and {col_title} ({label})\n"
                        f"(DA significant: FDR<{PVAL_CUTOFF}; {vote_title.lower()} significant: "
                        f">={min_hits}/{N_EMBEDDINGS} deployment embeddings with prob>0.5)"
                    ),
                    out_dir / f"cm_da_vs_{vote_slug}_{label}.png",
                )

            plot_confusion_grid_ordered(
                OMICS_NAMES,
                directions,
                permutation_ids_fn,
                TITLE_PERMUTATION,
                vote_ids_fn,
                vote_title,
                permutation_universe_fn,
                lambda row_title, col_title, vote_title=vote_title, min_hits=min_hits: (
                    f"Agreement between {row_title} and {col_title} ({label})\n"
                    f"(permutation testing significant: raw p<{PVAL_CUTOFF}; {vote_title.lower()} significant: "
                    f">={min_hits}/{N_EMBEDDINGS} deployment embeddings with prob>0.5)"
                ),
                out_dir / f"cm_permutation_testing_vs_{vote_slug}_{label}.png",
            )

        majority_ids_fn = lambda omics, direction: vote_ids(
            majority_hits, omics_ids, label, direction, omics, MAJORITY_MIN_HITS
        )
        unanimous_ids_fn = lambda omics, direction: vote_ids(
            majority_hits, omics_ids, label, direction, omics, UNANIMOUS_MIN_HITS
        )
        plot_confusion_grid_ordered(
            OMICS_NAMES,
            directions,
            majority_ids_fn,
            TITLE_MAJORITY,
            unanimous_ids_fn,
            TITLE_UNANIMOUS,
            lambda omics: omics_ids[omics],
            lambda row_title, col_title: (
                f"Agreement between {row_title} and {col_title} ({label})\n"
                f"(majority vote significant: >={MAJORITY_MIN_HITS}/{N_EMBEDDINGS}; unanimous vote significant: "
                f">={UNANIMOUS_MIN_HITS}/{N_EMBEDDINGS} deployment embeddings with prob>0.5)"
            ),
            out_dir / f"cm_majority_vote_vs_unanimous_vote_{label}.png",
        )


if __name__ == "__main__":
    main()
