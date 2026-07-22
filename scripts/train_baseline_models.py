"""
Author: Keenan Manpearl
Date: 2024-09-09

This script trains a model using raw features
not an embedding space.

This study's specifics: raw feature counts come from
raw_data/microbe_metabolites_filtered_rank_normalized.csv, indexed by sample.
There is no joint sample-feature embedding here, so no feature-level
predictions are generated (see train_deployment_models.py for that).

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.dataset import Dataset
from src.classifier import LogisticRegressionClassifier
from train_deployment_models import SEED, MAX_ITER, N_MODELS, SCORING


# time_labels.tsv covers every sample (Base + End), so it's validated
# against the full samples.txt; diet_labels.tsv only covers endpoint samples
# (diet group isn't meaningful at baseline), so it has no samples_path -
# Dataset.from_label_tsv just trusts diet_labels.tsv's own rows instead
SAMPLES_PATH = {"time": "data/nodes/samples.txt", "diet": None}


def _load_raw_omics_dataset(label_name: str) -> Dataset:
    """
    Build a Dataset for the given classifier target using raw rank-normalized
    -omics counts, with all 5 folds combined and used as PredefinedSplit CV folds.
    """
    fp = "raw_data/microbe_metabolites_filtered_rank_normalized.csv"
    raw = pd.read_csv(fp, index_col="sample").fillna(0)
    label_tsv = f"data/{label_name}_labels.tsv"

    return Dataset.from_label_tsv(
        label_name=label_name,
        feature_table=raw,
        label_tsv=label_tsv,
        split_tsv="data/node_splits.tsv",
        samples_path=SAMPLES_PATH[label_name],
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
        time_clf.write_results(f, n_folds=time_dataset.cv_folds.get_n_splits())
        f.write("\n")
        f.write("============== Diet Classification ==============\n")
        diet_clf.write_results(f, n_folds=diet_dataset.cv_folds.get_n_splits())


if __name__ == "__main__":
    main()
