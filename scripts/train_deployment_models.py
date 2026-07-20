"""
Author: Keenan Manpearl
Date: 2024-09-09

This script trains a classifier using all availble samples per fold.
Training/evaluations are done using samples (infants)
and predictions are made for features (microbes and metabolites).

This study's specifics: samples and features are jointly embedded via
node2vec+, with which rows are samples vs. features given explicitly by
data/nodes/samples.txt and data/nodes/{microbes,metabolites}.txt (not a
naming-convention heuristic); the "best" embedding for a given (p, q, g) is
expected to already be cached at data/emb/emb_p_{p}_q_{g}.tsv.gz. Labels
come from data/{label_name}_labels.tsv and CV fold assignment from
data/node_splits.tsv (see sample_labels.py and generate_splits.py to
regenerate them).

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from argparse import ArgumentParser
from datetime import datetime
import warnings
import os

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.dataset import Dataset
from src.classifier import LogisticRegressionClassifier
from src.zscoring import FeatureZScorer

warnings.filterwarnings("ignore", category=ConvergenceWarning)


SEED = 22
MAX_ITER = 100
N_MODELS = 500
N_FOLDS = 5
SCORING = ["f1", "balanced_accuracy", "accuracy"]
PRED_COLUMNS = {"time": ["baseline", "endpoint"], "diet": ["dairy", "meat"]}


def setup_output_dir(out_dir: str | None) -> str:
    """
    create output directory if it does not exist
    """
    if out_dir is None:
        current_date = datetime.now().strftime("%Y-%m-%d")
        out_dir = f"results/best_{current_date}"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _load_embedding_dataset(p: float, q: float, g: int, label_name: str) -> Dataset:
    """
    Build a Dataset for the given classifier target using the "best" node2vec+
    embedding for (p, q, g), with all 5 folds combined and used as
    PredefinedSplit CV folds. There is no separate held-out test set: the
    same data is used for CV search and the final full-data fit.
    """
    emb_file = f"data/emb/emb_p_{p}_q_{q}_g_{g}.tsv.gz"
    emb = pd.read_csv(emb_file, sep="\t", index_col=0)
    label_tsv = f"data/{label_name}_labels.tsv"

    return Dataset.from_label_tsv(
        label_name=label_name,
        feature_table=emb,
        label_tsv=label_tsv,
        split_tsv="data/node_splits.tsv",
        samples_path="data/nodes/samples.txt",
        feature_paths=["data/nodes/microbes.txt", "data/nodes/metabolites.txt"],
    )


def main(p: float, q: float, g: int, out_dir: str, tag: str) -> None:
    """
    main function to train models to evaluate embedding space using CV
    and extract final weights from full model
    """
    print(tag)

    time_dataset = _load_embedding_dataset(p, q, g, "time")
    diet_dataset = _load_embedding_dataset(p, q, g, "diet")

    time_clf = LogisticRegressionClassifier(
        label_name="time",
        pred_columns=PRED_COLUMNS["time"],
        seed=SEED,
        cv_max_iter=MAX_ITER,
        n_iter_search=N_MODELS,
        scoring=SCORING,
        refit="f1",
    )
    diet_clf = LogisticRegressionClassifier(
        label_name="diet",
        pred_columns=PRED_COLUMNS["diet"],
        seed=SEED,
        cv_max_iter=MAX_ITER,
        n_iter_search=N_MODELS,
        scoring=SCORING,
        refit="f1",
    )
    time_result = time_clf.run_deployment(time_dataset)
    diet_result = diet_clf.run_deployment(diet_dataset)

    with open(f"{out_dir}/{tag}_logging.txt", "w") as f:
        f.write("============== Node2Vec Parameters ==============\n")
        f.write(f"p: {p}\n")
        f.write(f"q: {q}\n")
        f.write(f"gamma: {g}\n")
        f.write("\n")
        f.write("============== Timepoint Classification ==============\n")
        time_clf.write_results(f, n_folds=N_FOLDS)
        f.write("\n")
        f.write("============== Diet Classification ==============\n")
        diet_clf.write_results(f, n_folds=N_FOLDS)

    zscorer = FeatureZScorer.from_files(
        {
            "microbes": "data/nodes/microbes.txt",
            "metabolites": "data/nodes/metabolites.txt",
        }
    )
    for clf, result, model_type in [
        (time_clf, time_result, "time"),
        (diet_clf, diet_result, "diet"),
    ]:
        clf.save(f"{out_dir}/{tag}_{model_type}_model.pkl")
        clf.save_weights(f"{out_dir}/{tag}_{model_type}_model_weights.txt")
        if result.feature_predictions is not None:
            result.feature_predictions.to_csv(
                f"{out_dir}/{tag}_{model_type}_feature_predictions.tsv", sep="\t"
            )
            zscorer.score_and_save(result.feature_predictions, out_dir, tag, model_type)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--out",
        help="output directory to save files",
        required=False,
        type=str,
        default=None,
    )
    args = parser.parse_args()
    out_dir = setup_output_dir(args.out)

    models = {
        "wcksnlsg": {"p": 19.0, "q": 9.122152261131532, "g": 1},
        "ai9n4jxs": {"p": 0.8055551041134607, "q": 0.1, "g": 1},
        "7o4yga2v": {"p": 0.5, "q": 1.895944090041435, "g": 1},
        "21tdsqsa": {"p": 1.0795506927238254, "q": 8.383911078685804, "g": 1},
        "q2gzu1o3": {"p": 19.0, "q": 8.483911078685804, "g": 2},
        "8lofhbbf": {"p": 7.305688086564288, "q": 7.517332462471247, "g": 2},
        "qb4y98x0": {"p": 5.5, "q": 9.010757520712524, "g": 1},
    }

    for model, params in models.items():
        main(p=params["p"], q=params["q"], g=params["g"], out_dir=out_dir, tag=model)
