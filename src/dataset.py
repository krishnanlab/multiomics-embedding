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

  This is also where nested cross-validation happens, and where the two
  supported cv_folds shapes diverge in role:

  - A PredefinedSplit (built from a split tsv - see from_label_tsv) is a set
    of OUTER folds that you define ahead of time, once, outside this library
    - typically stratified by class label and any other property you care
      about (e.g. age, site, batch). Nothing in src/ builds these folds for
      you; that choice belongs to the caller/study, since only you know what
      needs to be balanced across folds. When Dataset.cv_folds is a
      PredefinedSplit, RandomizedSearchCV validates directly against those
      outer folds - one level of CV, no nesting.
  - An int (e.g. n_folds) tells scikit-learn to run its own INNER k-fold CV
    automatically. Because LogisticRegression is a classifier, scikit-learn's
    default behavior for an int cv is StratifiedKFold - so this inner CV is
    automatically label-stratified, with no extra configuration needed.

  The nested design used by the sweep pipeline (see Classifier.run_sweep)
  combines both: for each user-defined OUTER fold (one PredefinedSplit value
  held out as a genuine test set), an INNER int cv_folds drives an automatic
  stratified hyperparameter search over just that fold's training data. You
  supply the outer folds; the library handles the inner ones.
- test_data / test_labels: optional, same shape rules as train_data/train_labels,
  for pipelines that evaluate against a genuinely held-out split.
- feature_matrix: optional DataFrame indexed by feature ID (e.g. a gene,
  microbe, or metabolite name), living in the same column/embedding space as
  train_data. Only meaningful for data sources that jointly embed samples and
  features (e.g. a node2vec+ embedding of a sample-feature graph); leave as
  None for data sources with no such shared space (e.g. raw feature counts).

A label tsv + split tsv (see from_label_tsv) are an alternative to building
a labels Series yourself: two tab-separated files, both indexed by node (the
sample ID). The label tsv has one other column, label (binary 0/1). The
split tsv has one other column, split (an integer CV fold in which that
node is held out as test; a sentinel value - -1 by default - marks nodes
that are never held out, which still count as training data for every
fold, just never evaluated on). Splitting them apart means one split tsv
can be shared across every label type for a study, instead of duplicating
fold assignment into every label file.

from_label_tsv also accepts samples_path and feature_paths: plain text
files, one ID per line, telling it which of feature_table's rows are
training samples and which are extra feature-prediction rows - replacing
any naming-convention heuristic (e.g. an "MD" prefix) with an explicit,
checked list. Every sample in samples_path must have both a label and a row
in feature_table, and every feature in feature_paths must have a row in
feature_table - otherwise from_label_tsv raises ValueError. Rows in
feature_table or the label tsv that aren't listed in samples_path/
feature_paths trigger a UserWarning rather than an error, in case that's
unintentional.

"""

import warnings
import dataclasses
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.model_selection import BaseCrossValidator, PredefinedSplit


def read_ids(path: str) -> "set[str]":
    """read a newline-separated list of IDs (samples or features) from a text file"""
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def require_present(
    ids: "set[str]", available: "pd.Index | set[str]", kind: str, requirement: str
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
    available: "pd.Index | set[str]", accounted_for: "set[str]", description: str
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

    @classmethod
    def from_label_tsv(
        cls,
        label_name: str,
        feature_table: pd.DataFrame,
        label_tsv: str,
        split_tsv: str,
        samples_path: "str | None" = None,
        feature_paths: "list[str] | None" = None,
        no_split_sentinel: int = -1,
    ) -> "Dataset":
        """
        Build a Dataset from a label tsv (columns: node, label) joined against
        a split tsv (columns: node, split) - see the module docstring for the
        exact format, and for what samples_path/feature_paths do. Rows whose
        split equals no_split_sentinel (never held out in any fold) are
        dropped, since PredefinedSplit has no meaningful fold to assign them
        to; cv_folds is a PredefinedSplit built from the remaining rows'
        split column.
        """
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

        features: "set[str]" = set()
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
