"""
Author: Keenan Manpearl
Date: 2026-07-20

Generic container for a binary-classification train split (optionally a
held-out test split), plus optionally a matrix of extra rows to predict on
once a classifier is fit.

Fields
------
- train_data/train_labels: aligned rows (DataFrame/ndarray) and 0/1 labels,
  same order.
- cv_folds: int (auto stratified k-fold - used as INNER CV during
  hyperparameter tuning) or a sklearn CV splitter like PredefinedSplit
  (fixed OUTER folds - used for deployment models, no tuning). The sweep
  pipeline (see Classifier.run_sweep) nests both: PredefinedSplit picks each
  outer test fold, an int drives the inner search within it.
- test_data/test_labels: optional held-out split (deployment models have
  none).
- feature_matrix: optional, indexed by feature ID, same column space as
  train_data - rows to generate predictions for. None if not applicable.

from_label_tsv builds a Dataset from a label tsv (columns: node, label) and
a split tsv (columns: node, split; split == no_split_sentinel means "never
held out" - dropped, since PredefinedSplit needs a real fold for every row).
samples_path/feature_paths (plain text, one ID per line) say which
feature_table rows are training samples vs. feature-prediction rows: a
sample/feature missing its label or feature_table row raises ValueError; a
feature_table/label row not claimed by either path just warns.

"""

import warnings
import dataclasses
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.model_selection import BaseCrossValidator, PredefinedSplit


def read_ids(path: str) -> set[str]:
    """read a newline-separated list of IDs (samples or features) from a text file"""
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def require_present(
    ids: set[str], available: pd.Index | set[str], kind: str, requirement: str
) -> None:
    """raise ValueError listing any of ids missing from available"""
    missing = ids - set(available)
    if missing:
        example = ", ".join(sorted(missing)[:10])
        suffix = ", ..." if len(missing) > 10 else ""
        raise ValueError(
            f"{len(missing)} {kind}(s) have no {requirement}: {example}{suffix}"
        )


def warn_if_unaccounted(
    available: pd.Index | set[str], accounted_for: set[str], description: str
) -> None:
    """warn listing any entries of available missing from accounted_for"""
    extra = set(available) - accounted_for
    if extra:
        example = ", ".join(sorted(extra)[:10])
        suffix = ", ..." if len(extra) > 10 else ""
        warnings.warn(
            f"{len(extra)} {description} are not listed in any provided "
            f"sample/feature path: {example}{suffix}"
        )


@dataclass
class Dataset:
    """
    label_name is purely a descriptive tag (e.g. "diet", "disease_status") used
    by callers for output naming - it carries no branching logic here.
    """

    label_name: str
    train_data: pd.DataFrame | np.ndarray
    train_labels: np.ndarray
    cv_folds: int | BaseCrossValidator
    test_data: pd.DataFrame | np.ndarray | None = None
    test_labels: np.ndarray | None = None
    feature_matrix: pd.DataFrame | None = None

    def features_to_predict(self) -> pd.DataFrame | None:
        """rows to generate predictions for; None if this data source doesn't support it"""
        return self.feature_matrix

    def with_shuffled_labels(self, rng: np.random.Generator) -> "Dataset":
        """Copy with train_labels randomly permuted - the permutation-test extension point."""
        return dataclasses.replace(
            self, train_labels=rng.permutation(self.train_labels)
        )

    @classmethod
    def from_tables(
        cls,
        label_name: str,
        feature_table: pd.DataFrame,
        labels: pd.Series,
        cv_folds: int | BaseCrossValidator,
        feature_matrix: pd.DataFrame | None = None,
    ) -> "Dataset":
        """Build a Dataset from an already-loaded feature table and labels, aligned by index."""
        return cls(
            label_name=label_name,
            train_data=feature_table.loc[labels.index],
            train_labels=labels.to_numpy(),
            cv_folds=cv_folds,
            feature_matrix=feature_matrix,
        )

    @classmethod
    def from_label_tsv(
        cls,
        label_name: str,
        feature_table: pd.DataFrame,
        label_tsv: str,
        split_tsv: str,
        samples_path: str | None = None,
        feature_paths: list[str] | None = None,
        no_split_sentinel: int = -1,
    ) -> "Dataset":
        """Build a Dataset from a label tsv + split tsv - see module docstring for format."""
        labels = pd.read_csv(label_tsv, sep="\t", index_col="node")["label"]
        splits = pd.read_csv(split_tsv, sep="\t", index_col="node")["split"]

        # validate against the raw label file, before the sentinel-fold filter below
        # drops nodes with no CV fold - those still have labels, just no fold to train/test on
        samples = (
            read_ids(samples_path) if samples_path is not None else set(labels.index)
        )
        require_present(samples, labels.index, "sample", "a label")
        require_present(samples, feature_table.index, "sample", "a row in feature_table")
        warn_if_unaccounted(labels.index, samples, "label_tsv rows")

        joined = pd.concat([labels, splits], axis=1, join="inner")
        joined = joined[joined.index.isin(samples)]
        joined = joined[joined["split"] != no_split_sentinel]

        features: set[str] = set()
        for path in feature_paths or []:
            features |= read_ids(path)
        if feature_paths is not None:
            require_present(
                features, feature_table.index, "feature", "a row in feature_table"
            )

        if samples_path is not None or feature_paths is not None:
            warn_if_unaccounted(
                feature_table.index, samples | features, "feature_table rows"
            )

        return cls.from_tables(
            label_name=label_name,
            feature_table=feature_table,
            labels=joined["label"],
            cv_folds=PredefinedSplit(joined["split"]),
            feature_matrix=(
                feature_table[feature_table.index.isin(features)]
                if feature_paths is not None
                else None
            ),
        )
