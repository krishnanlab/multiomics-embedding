"""
A generic container for a binary-classification train split (and, optionally,
a held-out test split), plus optionally a matrix of extra rows to generate
predictions for once a classifier is trained on it.

This class has no knowledge of any particular study's file paths, column
names, or label semantics - those belong in the scripts that build a Dataset,
not here. Bring your own multi-omics dataset by loading it however you like
and constructing a Dataset from the resulting tables.

Required data formats
----------------------
- train_data: a DataFrame or ndarray, one row per sample, one column per
  feature/embedding dimension. Row order must be aligned with train_labels.
- train_labels: a 1-D array-like of binary (0/1) labels, same length and
  order as train_data's rows.
- cv_folds: either an int (plain k-fold with that many folds) or any
  scikit-learn cross-validation splitter object (e.g. PredefinedSplit) that
  can be passed directly as RandomizedSearchCV's `cv` argument.
- test_data / test_labels: optional, same shape rules as train_data/train_labels,
  for pipelines that evaluate against a genuinely held-out split.
- feature_matrix: optional DataFrame indexed by feature ID (e.g. a gene,
  microbe, or metabolite name), living in the same column/embedding space as
  train_data. Only meaningful for data sources that jointly embed samples and
  features (e.g. a node2vec+ embedding of a sample-feature graph); leave as
  None for data sources with no such shared space (e.g. raw feature counts).

"""

import dataclasses
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.model_selection import BaseCrossValidator


@dataclass
class Dataset:
    """
    label_name is purely a descriptive tag (e.g. "diet", "disease_status") used
    by callers for output naming - it carries no branching logic here.
    """

    label_name: str
    train_data: "pd.DataFrame | np.ndarray"
    train_labels: np.ndarray
    cv_folds: "int | BaseCrossValidator"
    test_data: "pd.DataFrame | np.ndarray | None" = None
    test_labels: "np.ndarray | None" = None
    feature_matrix: "pd.DataFrame | None" = None

    def features_to_predict(self) -> "pd.DataFrame | None":
        """rows to generate predictions for; None if this data source doesn't support it"""
        return self.feature_matrix

    def with_shuffled_labels(self, rng: np.random.Generator) -> "Dataset":
        """
        Return a copy of this dataset with train_labels randomly permuted.
        This is the extension point for permutation testing: retrain a
        Classifier on the shuffled copy, predict features, and z-score the
        result to build a null distribution, without changing Dataset or
        Classifier at all.
        """
        return dataclasses.replace(
            self, train_labels=rng.permutation(self.train_labels)
        )

    @classmethod
    def from_tables(
        cls,
        label_name: str,
        feature_table: pd.DataFrame,
        labels: pd.Series,
        cv_folds: "int | BaseCrossValidator",
        feature_matrix: "pd.DataFrame | None" = None,
    ) -> "Dataset":
        """
        Build a Dataset from an already-loaded feature table and labels.
        feature_table and labels are aligned by index (feature_table.loc[labels.index]),
        so labels does not need to already be in the same row order as feature_table.
        """
        return cls(
            label_name=label_name,
            train_data=feature_table.loc[labels.index],
            train_labels=labels.to_numpy(),
            cv_folds=cv_folds,
            feature_matrix=feature_matrix,
        )
