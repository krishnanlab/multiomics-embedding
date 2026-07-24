"""
Author: Keenan Manpearl
Date: 2026-07-24

Permutation testing for the per-embedding feature-consensus z-score pipeline
(see src/zscoring.py's FeatureZScorer): fit each embedding's already-trained
deployment hyperparameters, z-score its feature predictions, and call a
feature a "hit" when |z| >= threshold. The observed consensus score per
feature is the fraction of embeddings that hit; label-permutation trials
build a null distribution for empirical p-values/BH q-values (see combine()).

No hyperparameter search: each embedding's best_params is given directly
(dict, or a path to its deployment run's ..._logging.txt - see
_parse_best_params); every trial is just fit_full on shuffled labels.

run_trial() draws ONE shared label permutation, applied identically to every
embedding, shuffled WITHIN each outer fold only (preserves each fold's
pos/neg count).

save()/load() pickle a full PermutationTest (Datasets + classifiers +
best_params) so worker processes (scripts/run_permutation_batch.py) can
load() and start computing trials immediately.

"""

import dataclasses
import pickle
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from sklearn.linear_model import LogisticRegression

from src.classifier import DEFAULT_PARAM_DISTRIBUTIONS, LogisticRegressionClassifier
from src.dataset import Dataset, validate_pred_columns
from src.zscoring import FeatureZScorer


def _parse_best_params(log_path: str) -> dict:
    """Parse "best <param>: <value>" lines (see write_results) into a hyperparameter dict, coercing to float where possible."""
    params = {}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("best "):
                continue
            name, _, value = line[len("best "):].partition(":")
            value = value.strip()
            try:
                value = float(value)
            except ValueError:
                pass
            params[name.strip()] = value
    return params


_VALID_LR_PARAMS = set(LogisticRegression().get_params().keys())
# keys every DEFAULT_PARAM_DISTRIBUTIONS combo supplies (l1_ratio is
# elasticnet-only, so not required)
_EXPECTED_BEST_PARAM_KEYS = set.intersection(
    *(set(combo.keys()) for combo in DEFAULT_PARAM_DISTRIBUTIONS)
)


def _validate_best_params(params: dict) -> None:
    """raise on an unknown LogisticRegression param key; warn if suspiciously incomplete"""
    unknown = set(params.keys()) - _VALID_LR_PARAMS
    if unknown:
        raise ValueError(
            f"best_params has unknown LogisticRegression parameter(s): "
            f"{sorted(unknown)} - full dict: {params}"
        )
    missing = _EXPECTED_BEST_PARAM_KEYS - set(params.keys())
    if missing:
        warnings.warn(f"best_params is missing expected key(s) {sorted(missing)}: {params}")


@dataclass
class ConsensusResult:
    """Per-feature observed (unpermuted) consensus scores/direction from PermutationTest.observed()."""

    scores: pd.Series  # index=feature ID, value in [0,1] = fraction of embeddings with |z| >= threshold
    direction: "pd.Series"  # index=feature ID, value = which class the real fit leans toward
    z_scores: pd.DataFrame  # index=feature ID, columns=embedding position 0..n-1, z(reference_class)


class PermutationTest:
    """Builds one Dataset per embedding from its known best_params; computes observed consensus + null trials. See module docstring."""

    def __init__(
        self,
        embeddings: list[pd.DataFrame],
        label_name: str,
        label_tsv: str,
        split_tsv: str,
        zscorer: FeatureZScorer,
        best_params: "list[dict | str]",
        *,
        pred_columns: "list[str] | None" = None,
        reference_class: "str | int | None" = None,
        threshold: float = 2.0,
        samples_path: "str | None" = None,
        feature_paths: "list[str] | None" = None,
        seed: int = 42,
        fit_max_iter: int = 1000,
    ) -> None:
        if not embeddings:
            raise ValueError("embeddings must be non-empty")
        if len(best_params) != len(embeddings):
            raise ValueError(
                f"got {len(embeddings)} embeddings but {len(best_params)} best_params "
                "entries - need exactly one per embedding"
            )
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        validate_pred_columns(pred_columns)
        self.embeddings = embeddings
        self.label_name = label_name
        self.label_tsv = label_tsv
        self.split_tsv = split_tsv
        self.zscorer = zscorer
        self.best_params = [
            _parse_best_params(bp) if isinstance(bp, str) else bp for bp in best_params
        ]
        for params in self.best_params:
            _validate_best_params(params)
        self.pred_columns = pred_columns
        self.reference_class = (
            reference_class
            if reference_class is not None
            else (pred_columns[-1] if pred_columns is not None else 1)
        )
        # binary-only, like the rest of this codebase (LogisticRegressionClassifier)
        self.other_class = (
            next(c for c in pred_columns if c != self.reference_class)
            if pred_columns is not None
            else (0 if self.reference_class == 1 else 1)
        )
        self.threshold = threshold
        self.samples_path = samples_path
        self.feature_paths = feature_paths
        self.seed = seed
        self.fit_max_iter = fit_max_iter

        self.datasets = self._build_datasets()
        self.classifiers = [self._make_classifier() for _ in self.datasets]

        # every dataset shares the same row order (all built from the same
        # label_tsv/split_tsv/samples_path), so any one of them's fold
        # assignment applies to all - see run_trial
        fold_labels = self.datasets[0].cv_folds.test_fold
        self._fold_groups = {
            fold: np.flatnonzero(fold_labels == fold) for fold in np.unique(fold_labels)
        }

    def _build_datasets(self) -> list[Dataset]:
        return [
            Dataset.from_label_tsv(
                label_name=self.label_name,
                feature_table=emb,
                label_tsv=self.label_tsv,
                split_tsv=self.split_tsv,
                samples_path=self.samples_path,
                feature_paths=self.feature_paths,
            )
            for emb in self.embeddings
        ]

    def _make_classifier(self) -> LogisticRegressionClassifier:
        # cv_max_iter/n_iter_search are required by the constructor but
        # never actually used - cv_search is never called here, only
        # fit_full (which falls back to cv_max_iter only when
        # fit_max_iter is None, so pointing both at fit_max_iter keeps
        # that fallback inert regardless)
        return LogisticRegressionClassifier(
            label_name=self.label_name,
            pred_columns=self.pred_columns,
            seed=self.seed,
            cv_max_iter=self.fit_max_iter,
            n_iter_search=1,
            fit_max_iter=self.fit_max_iter,
        )

    def _reference_z(self, preds: pd.DataFrame) -> pd.Series:
        """z(reference_class) for every feature, combined across every FeatureZScorer subset"""
        scored = self.zscorer.score(preds)
        return pd.concat([df[self.reference_class] for df in scored.values()])

    def observed(self) -> ConsensusResult:
        """Real (unpermuted) per-feature consensus score: fit_full on true labels, fraction of embeddings hitting |z| >= threshold, plus lean direction."""
        z_scores = {}
        for i, (dataset, clf, params) in enumerate(
            zip(self.datasets, self.classifiers, self.best_params)
        ):
            clf.fit_full(dataset, params=params)
            preds = clf.predict_features(dataset)
            z_scores[i] = self._reference_z(preds)
        z_df = pd.DataFrame(z_scores)

        scores = (z_df.abs() >= self.threshold).mean(axis=1)
        direction = pd.Series(
            np.where(z_df.mean(axis=1) >= 0, self.reference_class, self.other_class),
            index=z_df.index,
        )
        return ConsensusResult(scores=scores, direction=direction, z_scores=z_df)

    def run_trial(self, rng: np.random.Generator) -> pd.Series:
        """One null trial: shared within-fold label permutation applied to every embedding, returns per-feature hit fraction. No search - reuses best_params."""
        permuted = self.datasets[0].train_labels.copy()
        for idx in self._fold_groups.values():
            permuted[idx] = rng.permutation(permuted[idx])

        hits = {}
        for i, (dataset, clf, params) in enumerate(
            zip(self.datasets, self.classifiers, self.best_params)
        ):
            shuffled = dataclasses.replace(dataset, train_labels=permuted)
            clf.fit_full(shuffled, params=params)
            preds = clf.predict_features(shuffled)
            z = self._reference_z(preds)
            hits[i] = z.abs() >= self.threshold
        return pd.DataFrame(hits).mean(axis=1)

    def run_batch(self, n_permutations: int, seed: int) -> pd.DataFrame:
        """n_permutations independent run_trial() calls - one column per trial, index=feature ID"""
        rng = np.random.default_rng(seed)
        trials = {i: self.run_trial(rng) for i in range(n_permutations)}
        return pd.DataFrame(trials)

    def save(self, path: str) -> None:
        """pickle this PermutationTest, so worker processes can load() it without rebuilding it"""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "PermutationTest":
        """unpickle a PermutationTest saved by save()"""
        with open(path, "rb") as f:
            return pickle.load(f)


def combine(
    observed_scores: pd.Series,
    direction: pd.Series,
    null_scores: pd.DataFrame,
    feature_groups: "dict[str, list[str]] | None" = None,
) -> pd.DataFrame:
    """
    Combine observed per-feature consensus scores against null scores
    (index=feature, one column per trial) into empirical one-sided p-values
    (p = (1 + count(null >= observed)) / (1 + n_permutations), avoids p=0)
    and BH q-values. at_permutation_floor flags features at the minimum
    achievable p-value (no null trial ever met/exceeded observed) - a signal
    to run more permutations, not that the result is final.

    feature_groups (e.g. {"microbes": [...], "metabolites": [...]}), if
    given, runs BH separately per group instead of across all features -
    appropriate since groups have different counts/base rates.
    """
    n_permutations = null_scores.shape[1]
    exceed_counts = null_scores.ge(observed_scores, axis=0).sum(axis=1)
    p_values = (1 + exceed_counts) / (1 + n_permutations)

    if feature_groups is None:
        q_values = pd.Series(
            false_discovery_control(p_values.to_numpy(), method="bh"), index=p_values.index
        )
    else:
        q_values = pd.Series(index=p_values.index, dtype=float)
        for features in feature_groups.values():
            idx = p_values.index.intersection(features)
            q_values.loc[idx] = false_discovery_control(
                p_values.loc[idx].to_numpy(), method="bh"
            )
        uncovered = q_values.index[q_values.isna()]
        if len(uncovered):
            example = ", ".join(str(f) for f in sorted(uncovered)[:10])
            suffix = ", ..." if len(uncovered) > 10 else ""
            warnings.warn(
                f"{len(uncovered)} feature(s) are not covered by any "
                f"feature_groups entry, so their q_value is NaN: {example}{suffix}"
            )

    at_floor = exceed_counts == 0

    return pd.DataFrame(
        {
            "consensus_score": observed_scores,
            "direction": direction,
            "p_value": p_values,
            "q_value": q_values,
            "n_permutations": n_permutations,
            "at_permutation_floor": at_floor,
        }
    )
