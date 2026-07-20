"""
Author: Keenan Manpearl
Date: 2024-09-09

This script trains a model using raw features
not an embedding space.

This study's specifics: raw feature counts come from
data/raw/microbe_metabolites_filtered_rank_normalized.csv, indexed by sample.
There is no joint sample-feature embedding here, so no feature-level
predictions are generated (see train_deployment_models.py for that).

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import PredefinedSplit

from src.dataset import Dataset
from src.classifier import LogisticRegressionClassifier
from sample_labels import (
    load_test_indices,
    get_diet_indices,
    create_timepoint_labels,
    load_diet_labels,
)
from train_deployment_models import SEED, MAX_ITER, N_MODELS, N_FOLDS, SCORING


def _load_raw_omics_dataset(label_name: str) -> Dataset:
    """
    Build a Dataset for the given classifier target using raw rank-normalized
    -omics counts, with all 5 folds combined and used as PredefinedSplit CV folds.
    """
    time_indices = load_test_indices()
    diet_indices = get_diet_indices(time_indices)
    split_indices = time_indices if label_name == "time" else diet_indices

    fp = "data/raw/microbe_metabolites_filtered_rank_normalized.csv"
    raw = pd.read_csv(fp, index_col="sample").fillna(0)
    labels = (
        create_timepoint_labels(split_indices["nodes"])
        if label_name == "time"
        else load_diet_labels(split_indices["nodes"])
    )

    return Dataset.from_tables(
        label_name=label_name,
        feature_table=raw,
        labels=labels,
        cv_folds=PredefinedSplit(split_indices["run"]),
    )


def main() -> None:
    """
    main function to train models to evaluate embedding space using CV
    and extract final weights from full model
    """
    time_dataset = _load_raw_omics_dataset("time")
    diet_dataset = _load_raw_omics_dataset("diet")

    time_clf = LogisticRegressionClassifier(
        label_name="time",
        pred_columns=["baseline", "endpoint"],
        seed=SEED,
        cv_max_iter=MAX_ITER,
        n_iter_search=N_MODELS,
        scoring=SCORING,
        refit="f1",
    ).cv_search(time_dataset)
    diet_clf = LogisticRegressionClassifier(
        label_name="diet",
        pred_columns=["dairy", "meat"],
        seed=SEED,
        cv_max_iter=MAX_ITER,
        n_iter_search=N_MODELS,
        scoring=SCORING,
        refit="f1",
    ).cv_search(diet_dataset)

    with open("results/best/baseline_logging.txt", "w") as f:
        f.write("============== Timepoint Classification ==============\n")
        time_clf.write_results(f, n_folds=N_FOLDS)
        f.write("\n")
        f.write("============== Diet Classification ==============\n")
        diet_clf.write_results(f, n_folds=N_FOLDS)


if __name__ == "__main__":
    main()
