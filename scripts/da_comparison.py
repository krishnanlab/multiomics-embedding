"""
Author: Keenan Manpearl
Date: 2026-07-27

General-purpose helpers for comparing the classifier-based feature-consensus
permutation results (src/permutation.py's combine(), e.g.
results/permutations_time_*/combined.tsv) against a baseline per-feature
differential abundance table (raw_data/{microbiome,metabolites}.txt).

Built for notebooks/2026-07-27_time_consensus_vs_baseline_da.ipynb, but every
function takes feature-ID sets or column/op/threshold triples rather than
hardcoding a selection rule - so the same helpers work for a different
consensus_score cutoff, a different baseline FDR/effect-size cutoff, or the
"diet" comparison instead of "time", without editing this file.
"""

import glob
import os

import pandas as pd


BASELINE_ID_COL = "Feature Type/Library"
BASELINE_EFFECT_COL = "Fold Change"
BASELINE_GROUP_COL = "Group with higher relative abundace"
BASELINE_FDR_COL = "FDR "
BASELINE_PVALUE_COL = "Paired Wilcoxon rank sum test raw p-value"


def load_baseline_da(path: str, encoding: str = "latin-1") -> pd.DataFrame:
    """Load a raw_data/{microbiome,metabolites}.txt supplementary DA table.
    First line is a title, not the header - skip it. metabolites.txt has
    non-UTF8 bytes in its free-text annotation column, hence the latin-1
    default (microbiome.txt is plain ASCII and also reads fine with it)."""
    df = pd.read_csv(path, sep="\t", skiprows=1, encoding=encoding)
    df = df.rename(columns={BASELINE_FDR_COL: "FDR"})
    df = df.set_index(BASELINE_ID_COL)
    df.index.name = "feature"
    return df


def load_consensus_results(path: str) -> pd.DataFrame:
    """Load a permutation combine() output: index=feature, columns include
    consensus_score/direction/p_value/q_value/n_permutations/at_permutation_floor."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index.name = "feature"
    return df


def discover_consensus_modes(results_dir: str) -> "list[str]":
    """List the per-mode combined_<mode>.tsv files actually present in a
    permutation results dir (e.g. results/permutations_time_100000_b1000), by
    scanning the directory rather than hardcoding src/permutation.py's
    CONSENSUS_MODES names - those names have already changed once
    (hit_fraction -> hit_fraction_z, n_confident -> hit_fraction_prob), and a
    given results dir may lag or lead the current source (e.g. an old run
    still has combined_n_confident.tsv; a not-yet-rerun one is missing
    combined_hit_fraction_prob.tsv)."""
    paths = sorted(glob.glob(os.path.join(results_dir, "combined_*.tsv")))
    return [os.path.basename(p)[len("combined_"):-len(".tsv")] for p in paths]


def load_consensus_modes(results_dir: str, modes=None) -> "dict[str, pd.DataFrame]":
    """Load results/permutations_*_b1000/combined_<mode>.tsv for every mode
    (auto-discovered via discover_consensus_modes() if `modes` is None) - one
    DataFrame per mode, same schema as load_consensus_results() (score/
    direction/p_value/q_value/...). Lets a notebook loop over modes to
    compare them without hardcoding which modes exist."""
    if modes is None:
        modes = discover_consensus_modes(results_dir)
    return {mode: load_consensus_results(f"{results_dir}/combined_{mode}.tsv") for mode in modes}


def restrict_to_group(df: pd.DataFrame, node_ids_path: str) -> pd.DataFrame:
    """Restrict a feature-indexed table to the IDs listed in a
    data/nodes/*.txt file - e.g. split a combined consensus table (which
    covers microbes+metabolites together) into just one omics type."""
    with open(node_ids_path) as f:
        ids = {line.strip() for line in f if line.strip()}
    return df.loc[df.index.intersection(ids)]


_OPS = {
    ">": lambda s, t: s > t,
    ">=": lambda s, t: s >= t,
    "<": lambda s, t: s < t,
    "<=": lambda s, t: s <= t,
    "==": lambda s, t: s == t,
    "!=": lambda s, t: s != t,
}


def select_features(df: pd.DataFrame, column: str, op: str, threshold) -> set:
    """Generic threshold selector over any feature-indexed table: returns the
    set of feature IDs where `df[column] <op> threshold` holds. Covers
    consensus_score > 0, FDR < 0.05, q_value <= 0.1, etc. - to select on a
    derived quantity (e.g. |Fold Change|), pass a df with that column added."""
    if op not in _OPS:
        raise ValueError(f"unknown op {op!r} - must be one of {sorted(_OPS)}")
    return set(df.index[_OPS[op](df[column], threshold)])


def compare_feature_sets(set_a: set, set_b: set, name_a: str = "a", name_b: str = "b") -> dict:
    """Overlap summary between two feature-ID sets: shared/only-one-side
    members and counts, plus Jaccard index and the overlap coefficient
    (Szymkiewicz-Simpson: |both| / min(|a|,|b|)). Jaccard is diluted whenever
    the two sets are very different sizes (it divides by the union, which is
    dominated by the larger set); the overlap coefficient instead asks "what
    fraction of the smaller set is covered," which is the more useful number
    when one selection is much stricter than the other. Selection-method-agnostic."""
    both = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a
    union = set_a | set_b
    smaller = min(len(set_a), len(set_b))
    return {
        "both": both,
        f"only_{name_a}": only_a,
        f"only_{name_b}": only_b,
        "n_both": len(both),
        f"n_only_{name_a}": len(only_a),
        f"n_only_{name_b}": len(only_b),
        f"n_{name_a}": len(set_a),
        f"n_{name_b}": len(set_b),
        "jaccard": len(both) / len(union) if union else float("nan"),
        "overlap_coefficient": len(both) / smaller if smaller else float("nan"),
    }


def direction_agreement(
    baseline_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
    feature_ids,
    direction_map: dict,
) -> "tuple[pd.DataFrame, float]":
    """For features in both tables, compare baseline's "group with higher
    relative abundance" label against the consensus pipeline's `direction`
    column. `direction_map` translates consensus direction values (e.g.
    {"baseline": "5-month-old", "endpoint": "12-month-old"}) into baseline's
    label vocabulary, since the two tables name groups differently. Returns
    a per-feature table plus the fraction that agree."""
    ids = [f for f in feature_ids if f in baseline_df.index and f in consensus_df.index]
    baseline_group = baseline_df.loc[ids, BASELINE_GROUP_COL]
    consensus_dir = consensus_df.loc[ids, "direction"].map(direction_map)
    table = pd.DataFrame(
        {
            "baseline_group": baseline_group,
            "consensus_direction_mapped": consensus_dir,
        },
        index=ids,
    )
    table["agree"] = table["baseline_group"] == table["consensus_direction_mapped"]
    rate = table["agree"].mean() if len(table) else float("nan")
    return table, rate


def effect_size_summary(
    baseline_df: pd.DataFrame,
    feature_ids,
    effect_col: str = BASELINE_EFFECT_COL,
    absolute: bool = True,
) -> dict:
    """Median/mean/std/n of baseline effect size (default |Fold Change|, a
    log2 fold change per raw_data/README.md) over an arbitrary feature-ID
    set - e.g. found-by-both vs only-baseline vs only-consensus."""
    ids = [f for f in feature_ids if f in baseline_df.index]
    values = baseline_df.loc[ids, effect_col]
    if absolute:
        values = values.abs()
    return {"n": len(values), "median": values.median(), "mean": values.mean(), "std": values.std()}


def merge_tables(
    baseline_df: pd.DataFrame, consensus_df: pd.DataFrame, suffixes=("_baseline", "_consensus")
) -> pd.DataFrame:
    """Inner-join baseline and consensus tables on feature ID - for ad hoc
    scatter/correlation exploration beyond the fixed summaries above."""
    return baseline_df.join(consensus_df, how="inner", lsuffix=suffixes[0], rsuffix=suffixes[1])


def summarize_comparison(
    baseline_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
    baseline_ids: set,
    consensus_ids: set,
    direction_map: dict,
    name_baseline: str = "baseline",
    name_consensus: str = "consensus",
) -> dict:
    """Main entry point: overlap between two feature-ID selections, direction
    agreement on the overlap, and effect-size summaries for both/only-one-side.
    Swap in different baseline_ids/consensus_ids (any select_features() call,
    for either omics type) to re-run this exact comparison under different
    selection criteria without touching this function."""
    overlap = compare_feature_sets(baseline_ids, consensus_ids, name_baseline, name_consensus)
    dir_table, dir_rate = direction_agreement(baseline_df, consensus_df, overlap["both"], direction_map)
    effect = {
        "both": effect_size_summary(baseline_df, overlap["both"]),
        f"only_{name_baseline}": effect_size_summary(baseline_df, overlap[f"only_{name_baseline}"]),
        f"only_{name_consensus}": effect_size_summary(baseline_df, overlap[f"only_{name_consensus}"]),
    }
    return {
        "overlap": overlap,
        "direction_agreement_table": dir_table,
        "direction_agreement_rate": dir_rate,
        "effect_size": effect,
    }
