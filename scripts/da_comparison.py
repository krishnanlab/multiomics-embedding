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

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity


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


def effect_size_percentiles(
    baseline_df: pd.DataFrame,
    feature_ids,
    reference_ids,
    effect_col: str = BASELINE_EFFECT_COL,
    absolute: bool = True,
    percentiles=(10, 25, 50, 75, 90),
) -> dict:
    """Where feature_ids' median |effect_col| falls within the reference_ids
    distribution of the same column - turns "boxplots looked different" into
    a concrete percentile rank (scipy.stats.percentileofscore), plus the
    reference distribution's own percentile cutpoints for context. E.g.
    reference_ids=all baseline-significant features, feature_ids=only the
    ones a consensus mode missed - reports how "typical" the missed set's
    effect size is relative to the significant population as a whole."""
    ref_ids = [f for f in reference_ids if f in baseline_df.index]
    ref_values = baseline_df.loc[ref_ids, effect_col]
    feat_ids = [f for f in feature_ids if f in baseline_df.index]
    feat_values = baseline_df.loc[feat_ids, effect_col]
    if absolute:
        ref_values = ref_values.abs()
        feat_values = feat_values.abs()
    cutpoints = {p: float(np.percentile(ref_values, p)) for p in percentiles} if len(ref_values) else {}
    feat_median = float(feat_values.median()) if len(feat_values) else float("nan")
    rank = (
        float(stats.percentileofscore(ref_values, feat_median))
        if len(ref_values) and len(feat_values)
        else float("nan")
    )
    return {
        "n_reference": len(ref_values),
        "n_features": len(feat_values),
        "reference_percentiles": cutpoints,
        "feature_median": feat_median,
        "feature_median_percentile_rank": rank,
    }


def _top_k_neighbor_ids(sim_row: np.ndarray, candidate_ids: list, k: int) -> list:
    """Indices of the k largest finite entries in sim_row (masked entries are
    -inf and never selected), mapped to candidate_ids - shared by
    knn_significance_fraction/cross_embedding_neighbor_overlap."""
    valid = np.isfinite(sim_row)
    k_eff = min(k, int(valid.sum()))
    if k_eff == 0:
        return []
    top_idx = np.argpartition(sim_row, -k_eff)[-k_eff:]
    return [candidate_ids[j] for j in top_idx]


def knn_significance_fraction(
    embedding_dfs: "list[pd.DataFrame]",
    query_ids,
    full_candidate_ids,
    same_omics_candidate_ids,
    reference_sig_ids,
    k: int = 15,
) -> "tuple[pd.Series, pd.Series]":
    """Cosine-similarity k-NN neighbor-significance fraction, averaged across
    multiple node2vec+ embeddings.

    embedding_dfs: list of DataFrames (one per embedding), each indexed by
    node ID (samples and features share the same space) with embedding-dim
    columns. For each embedding, builds a (len(query_ids), len(full_candidate_ids))
    cosine-similarity matrix (query features vs. the full candidate pool),
    then averages these matrices elementwise across all embeddings (keeps
    each embedding weighted equally regardless of vector norm - not the same
    as concatenating embeddings before computing similarity). A query
    feature that also appears in the candidate pool is never counted as its
    own neighbor (self-similarity masked to -inf before top-k selection).

    From that single averaged matrix, derives two per-query "fraction of the
    k nearest neighbors that fall in reference_sig_ids" results - one over
    the full candidate pool, one restricted to same_omics_candidate_ids (a
    subset of full_candidate_ids) - without recomputing cosine similarity a
    second time for the narrower version.

    Returns (full_pool_fractions, same_omics_fractions), each a pd.Series
    indexed by query_ids.
    """
    query_ids = list(query_ids)
    full_candidate_ids = list(full_candidate_ids)
    same_omics_mask = np.array([c in set(same_omics_candidate_ids) for c in full_candidate_ids])
    reference_sig_ids = set(reference_sig_ids)

    sim_sum = np.zeros((len(query_ids), len(full_candidate_ids)), dtype=np.float32)
    for emb_df in embedding_dfs:
        q = emb_df.loc[query_ids].to_numpy(dtype=np.float32)
        c = emb_df.loc[full_candidate_ids].to_numpy(dtype=np.float32)
        sim_sum += cosine_similarity(q, c).astype(np.float32)
    sim_avg = sim_sum / len(embedding_dfs)

    candidate_pos = {c: j for j, c in enumerate(full_candidate_ids)}
    for i, qid in enumerate(query_ids):
        j = candidate_pos.get(qid)
        if j is not None:
            sim_avg[i, j] = -np.inf

    def fraction_significant(sim_matrix: np.ndarray) -> pd.Series:
        out = {}
        for i, qid in enumerate(query_ids):
            neighbors = _top_k_neighbor_ids(sim_matrix[i], full_candidate_ids, k)
            out[qid] = (
                sum(n in reference_sig_ids for n in neighbors) / len(neighbors)
                if neighbors
                else float("nan")
            )
        return pd.Series(out)

    full_pool_fractions = fraction_significant(sim_avg)
    same_omics_sim = np.where(same_omics_mask[np.newaxis, :], sim_avg, -np.inf)
    same_omics_fractions = fraction_significant(same_omics_sim)
    return full_pool_fractions, same_omics_fractions


def cross_embedding_neighbor_overlap(
    embedding_dfs: "list[pd.DataFrame]", query_ids, candidate_ids, k: int = 15
) -> pd.Series:
    """For each feature in query_ids, finds its own k nearest neighbors (by
    cosine similarity, drawn from candidate_ids) SEPARATELY within each
    embedding in embedding_dfs (no averaging - the opposite of
    knn_significance_fraction), then returns the mean pairwise Jaccard
    overlap across all C(len(embedding_dfs),2) pairs of per-embedding
    neighbor sets: a per-feature "do the embeddings agree on who my
    neighbors are" consistency score. 1.0 = every embedding pair has an
    identical neighbor set; 0.0 = every pair is fully disjoint. A query
    feature is never counted as its own neighbor."""
    query_ids = list(query_ids)
    candidate_ids = list(candidate_ids)
    candidate_pos = {c: j for j, c in enumerate(candidate_ids)}

    per_embedding_neighbor_sets = []
    for emb_df in embedding_dfs:
        q = emb_df.loc[query_ids].to_numpy(dtype=np.float32)
        c = emb_df.loc[candidate_ids].to_numpy(dtype=np.float32)
        sim = cosine_similarity(q, c)
        for i, qid in enumerate(query_ids):
            j = candidate_pos.get(qid)
            if j is not None:
                sim[i, j] = -np.inf
        per_embedding_neighbor_sets.append(
            {qid: set(_top_k_neighbor_ids(sim[i], candidate_ids, k)) for i, qid in enumerate(query_ids)}
        )

    n_emb = len(embedding_dfs)
    pairs = [(a, b) for a in range(n_emb) for b in range(a + 1, n_emb)]
    scores = {}
    for qid in query_ids:
        overlaps = []
        for a, b in pairs:
            sa, sb = per_embedding_neighbor_sets[a][qid], per_embedding_neighbor_sets[b][qid]
            union = sa | sb
            overlaps.append(len(sa & sb) / len(union) if union else float("nan"))
        scores[qid] = float(np.nanmean(overlaps)) if overlaps else float("nan")
    return pd.Series(scores)


def pairwise_embedding_neighbor_agreement(
    embedding_dfs: "list[pd.DataFrame]", feature_ids, k: int = 15
) -> pd.DataFrame:
    """Embedding x embedding matrix: average Jaccard overlap between each
    pair of embeddings' k-NN neighbor sets, over every feature in
    feature_ids (not restricted to any comparison segment - a global
    characterization of how much two embeddings' local neighborhoods agree,
    given their different node2vec+ p/q/g hyperparameters). Diagonal is 1.0
    by definition."""
    feature_ids = list(feature_ids)

    neighbor_sets_per_embedding = []
    for emb_df in embedding_dfs:
        v = emb_df.loc[feature_ids].to_numpy(dtype=np.float32)
        sim = cosine_similarity(v, v)
        np.fill_diagonal(sim, -np.inf)
        neighbor_sets_per_embedding.append(
            [set(_top_k_neighbor_ids(sim[i], feature_ids, k)) for i in range(len(feature_ids))]
        )

    n_emb = len(embedding_dfs)
    mat = np.eye(n_emb)
    for a in range(n_emb):
        for b in range(a + 1, n_emb):
            overlaps = []
            for i in range(len(feature_ids)):
                sa, sb = neighbor_sets_per_embedding[a][i], neighbor_sets_per_embedding[b][i]
                union = sa | sb
                overlaps.append(len(sa & sb) / len(union) if union else float("nan"))
            mat[a, b] = mat[b, a] = float(np.nanmean(overlaps)) if overlaps else float("nan")
    return pd.DataFrame(mat, index=range(n_emb), columns=range(n_emb))


def load_per_embedding_predictions(paths: "list[str]", column: str = "endpoint") -> pd.DataFrame:
    """Join N deployment feature_predictions.tsv files (one per embedding,
    e.g. results/deployment_for_permutations/time_edges_p_*_feature_predictions.tsv -
    each a predict_proba table indexed by feature ID with one column per
    class) into a single (n_features, N) DataFrame of the given class's
    probability, one column per embedding, in the order paths are given."""
    columns = []
    for i, path in enumerate(paths):
        df = pd.read_csv(path, sep="\t", index_col=0)
        columns.append(df[column].rename(i))
    return pd.concat(columns, axis=1)


def per_feature_cross_embedding_cv(probs_df: pd.DataFrame) -> pd.Series:
    """Per-feature (row) coefficient of variation (std/mean) across embedding
    columns - high CV means the embeddings disagree on this feature's
    probability (some see it, most don't); low CV means they're consistent,
    whether confidently one way or only mildly."""
    return probs_df.std(axis=1) / probs_df.mean(axis=1)


def per_feature_degree(edges_path: str) -> pd.Series:
    """Per-feature node degree in a raw_data/edges.tsv-style bipartite
    sample-feature edge list (no header; columns: sample, feature, weight) -
    count of samples each feature has a non-missing measurement for. A
    data-quality/missingness covariate, since rank-normalization upstream
    drops absent measurements rather than zero-filling them."""
    edges = pd.read_csv(edges_path, sep="\t", header=None, names=["sample", "feature", "weight"])
    return edges.groupby("feature").size()


def per_feature_abundance_cv(abundance_csv_path: str, skip_cols=()) -> pd.Series:
    """Per-feature coefficient of variation (std/mean, NaN-safe) from a raw
    sample x feature abundance matrix (raw_data/{microbiome,metabolite}_data_
    for_differential_abundance.csv). skip_cols excludes any non-feature
    metadata columns (e.g. metabolites.csv's Group/Time/.../HCAZ, or
    microbiome.csv's Library index) before computing CV over the rest."""
    df = pd.read_csv(abundance_csv_path)
    df = df.drop(columns=[c for c in skip_cols if c in df.columns])
    return df.std(axis=0, skipna=True) / df.mean(axis=0, skipna=True)


def per_feature_endpoint_higher_fraction(
    abundance_csv_path: str,
    subject_col: str,
    time_col: str,
    baseline_time_value: str,
    endpoint_time_value: str,
    skip_cols=(),
) -> pd.DataFrame:
    """Per-feature, per-subject PAIRED change (endpoint - baseline), for
    subjects with both timepoints present - unlike per_feature_abundance_cv
    (marginal/unpaired), this reconstructs the actual pairing the baseline
    Wilcoxon signed-rank test uses (raw_data/{microbiome,metabolite}_data_
    for_differential_abundance.csv have per-subject IDs and a timepoint
    column). Returns a DataFrame with `frac_endpoint_higher` (fraction of
    paired subjects where endpoint>baseline, NaN-safe per feature) and
    `n_pairs` - lets a feature's paired-direction consistency be checked
    directly, since a rank test's significance can come from a bare majority
    of pairs agreeing, not necessarily most of them."""
    df = pd.read_csv(abundance_csv_path, low_memory=False)
    feature_cols = [c for c in df.columns if c not in skip_cols and c not in (subject_col, time_col)]
    df = df[df[time_col].isin([baseline_time_value, endpoint_time_value])]
    baseline = df[df[time_col] == baseline_time_value].drop_duplicates(subject_col).set_index(subject_col)[feature_cols]
    endpoint = df[df[time_col] == endpoint_time_value].drop_duplicates(subject_col).set_index(subject_col)[feature_cols]
    common = baseline.index.intersection(endpoint.index)
    baseline_num = baseline.loc[common].apply(pd.to_numeric, errors="coerce")
    endpoint_num = endpoint.loc[common].apply(pd.to_numeric, errors="coerce")
    diff = endpoint_num - baseline_num
    frac_higher = (diff > 0).sum(axis=0) / diff.notna().sum(axis=0)
    n_pairs = diff.notna().sum(axis=0)
    return pd.DataFrame({"frac_endpoint_higher": frac_higher, "n_pairs": n_pairs})


def pairwise_cosine_matrix(vectors: "list[np.ndarray]") -> pd.DataFrame:
    """Symmetric len(vectors) x len(vectors) cosine similarity matrix between
    a list of 1-D vectors THAT ALL LIVE IN THE SAME COORDINATE SYSTEM (e.g.
    several features' representations within one embedding). Do NOT use this
    to compare vectors from different node2vec+ embeddings directly (e.g.
    two different embeddings' fitted classifier weight vectors) - each
    embedding is an independently, arbitrarily rotated/scaled space, so raw
    cross-space vector-direction comparisons are close to meaningless (two
    unrelated random vectors from different high-dim spaces are already
    ~orthogonal - low cosine similarity there is a triviality, not a
    finding). To compare something across embeddings, first reduce each
    embedding to a well-defined scalar (e.g. per_feature_class_affinity's
    `affinity` column) and compare those instead."""
    mat = cosine_similarity(np.vstack(vectors))
    return pd.DataFrame(mat, index=range(len(vectors)), columns=range(len(vectors)))


def per_feature_class_affinity(
    embedding_df: pd.DataFrame, feature_ids, sample_labels: pd.Series
) -> pd.DataFrame:
    """Within ONE embedding (a valid, single coordinate system), each
    feature's mean cosine similarity to all class-1 samples minus its mean
    cosine similarity to all class-0 samples (sample_labels: 0/1, e.g.
    data/time_labels.tsv) - a geometric "which class does this feature look
    closer to" score that doesn't depend on any classifier fit. Unlike
    comparing raw weight vectors across embeddings (see pairwise_cosine_
    matrix's docstring), this reduces each embedding to a per-feature scalar
    first, so the resulting `affinity` values ARE meaningfully comparable
    across different embeddings' calls to this function. Returns columns
    `sim_class0`, `sim_class1`, `affinity` (=sim_class1 - sim_class0),
    indexed by feature_ids."""
    ids0 = [s for s in sample_labels.index[sample_labels == 0] if s in embedding_df.index]
    ids1 = [s for s in sample_labels.index[sample_labels == 1] if s in embedding_df.index]
    feature_ids = list(feature_ids)
    feat_vecs = embedding_df.loc[feature_ids].to_numpy()
    sim0 = cosine_similarity(feat_vecs, embedding_df.loc[ids0].to_numpy()).mean(axis=1)
    sim1 = cosine_similarity(feat_vecs, embedding_df.loc[ids1].to_numpy()).mean(axis=1)
    return pd.DataFrame({"sim_class0": sim0, "sim_class1": sim1, "affinity": sim1 - sim0}, index=feature_ids)


def sample_class_separation(embedding_df: pd.DataFrame, sample_labels: pd.Series) -> float:
    """Silhouette score of an embedding's SAMPLE rows (not features), given a
    per-sample binary label (e.g. data/time_labels.tsv) - how well-separated
    the two label classes are in that embedding's space on their own,
    independent of any classifier fit on top of it."""
    from sklearn.metrics import silhouette_score

    ids = [s for s in sample_labels.index if s in embedding_df.index]
    x = embedding_df.loc[ids].to_numpy()
    y = sample_labels.loc[ids].to_numpy()
    return float(silhouette_score(x, y))
