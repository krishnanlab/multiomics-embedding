"""
Author: Keenan Manpearl
Date: 2026-07-27

Helpers for notebooks/2026-07-27_zscore_prob_summary.ipynb: load the
per-embedding, per-feature raw probability / z-score matrices that sit
behind src/permutation.py's CONSENSUS_MODES (PermutationTest.observed()'s
ConsensusResult.probs/z_scores - fit_permutation_test.py only persists the
already-aggregated CONSENSUS_MODES scores in observed.tsv, not these), and
summarize their marginal distributions and cross-embedding agreement.
"""

import re

import numpy as np
import pandas as pd
from scipy import stats

from src.permutation import ConsensusResult, PermutationTest


def load_observed(fitted_state_path: str) -> ConsensusResult:
    """Load a fit_permutation_test.py fitted_state.pkl and recompute observed()
    to recover its per-embedding z_scores/probs matrices."""
    test = PermutationTest.load(fitted_state_path)
    return test.observed()


def split_group(df: pd.DataFrame, feature_ids) -> pd.DataFrame:
    """Restrict a feature-indexed matrix to one feature group's IDs (e.g.
    PermutationTest.zscorer.feature_lists['microbes'])."""
    ids = set(feature_ids)
    return df.loc[df.index.isin(ids)]


_PQG_RE = re.compile(r"p_([\d.]+)_q_([\d.]+)_g_(\d+)")


def embedding_labels(embedding_paths: "list[str]") -> "list[str]":
    """Short "p=.., q=.., g=.." labels parsed from emb_p_<p>_q_<q>_g_<g>.tsv(.gz)
    filenames, for readable axis/heatmap ticks instead of raw 0..6 positions."""
    labels = []
    for path in embedding_paths:
        m = _PQG_RE.search(path)
        if not m:
            raise ValueError(f"couldn't parse p/q/g from {path!r}")
        p, q, g = m.groups()
        labels.append(f"p={float(p):.2g}, q={float(q):.2g}, g={g}")
    return labels


def hit_mask(z_df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """boolean matrix, same shape as z_df: |z| >= threshold (each CONSENSUS_MODES
    mode's per-embedding "hit" call, e.g. hit_fraction/n_confident's z-based analog)"""
    return z_df.abs() >= threshold


def pairwise_corr(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """embedding x embedding correlation matrix over shared features (df's
    columns = embedding position)"""
    return df.corr(method=method)


def pairwise_agreement(hit_df: pd.DataFrame) -> pd.DataFrame:
    """embedding x embedding fraction of features where both embeddings'
    hit_mask() calls agree (both hit or both miss)"""
    cols = hit_df.columns
    mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            agree = (hit_df.iloc[:, i] == hit_df.iloc[:, j]).mean()
            mat.iloc[i, j] = mat.iloc[j, i] = agree
    return mat


def expected_agreement_if_independent(hit_df: pd.DataFrame) -> pd.DataFrame:
    """embedding x embedding chance-level agreement
    (P(agree) = p_i*p_j + (1-p_i)*(1-p_j)) two embeddings' hit calls would show
    if independent, given each one's own marginal hit rate - the baseline
    pairwise_agreement() should be compared against."""
    rates = hit_df.mean()
    cols = hit_df.columns
    mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pi, pj = rates.iloc[i], rates.iloc[j]
            mat.iloc[i, j] = mat.iloc[j, i] = pi * pj + (1 - pi) * (1 - pj)
    return mat


def distribution_summary(df: pd.DataFrame) -> pd.Series:
    """flatten a feature x embedding matrix and summarize its marginal
    distribution: n, mean, std, skew, excess kurtosis (normal = 0)"""
    values = df.to_numpy().ravel()
    values = values[~np.isnan(values)]
    return pd.Series(
        {
            "n": len(values),
            "mean": values.mean(),
            "std": values.std(),
            "skew": stats.skew(values),
            "kurtosis": stats.kurtosis(values),
        }
    )


def shape_summary(df: pd.DataFrame) -> pd.Series:
    """flatten a feature x embedding matrix and summarize its distribution's
    shape only (skew, excess kurtosis) - for FeatureZScorer's within-embedding
    z-scores, mean/std are ~0/~1 by construction and carry no information, so
    unlike distribution_summary() this omits them rather than reporting a
    guaranteed constant."""
    values = df.to_numpy().ravel()
    values = values[~np.isnan(values)]
    return pd.Series(
        {
            "n": len(values),
            "skew": stats.skew(values),
            "kurtosis": stats.kurtosis(values),
        }
    )


def hit_count_distribution(hit_df: pd.DataFrame) -> pd.Series:
    """per-feature count of embeddings that hit (0..n_embeddings), as a
    value_counts sorted by count - the raw histogram behind hit_fraction/n_confident"""
    return hit_df.sum(axis=1).value_counts().sort_index()
