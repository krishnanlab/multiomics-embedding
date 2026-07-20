"""
Author: Keenan Manpearl
Date: 2024-09-09

This script trains a node classifier to predict
baseline vs endpoint samples.

This study's specifics: node2vec+ embeddings are generated (or loaded from
cache) from data/edges.tsv, one per (p, q, gamma) combination, cached under
emb_cache/ (ephemeral per-trial output - not to be confused with data/emb/,
which holds the curated "best" embeddings used for deployment). Time/diet
labels come from data/{label_name}_labels.tsv and CV fold assignment from
data/node_splits.tsv (see sample_labels.py and generate_splits.py to
regenerate them).

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wandb

from src.dataset import Dataset
from src.classifier import LogisticRegressionClassifier
from src.embedding import load_or_create_embedding
from sample_labels import load_time_labels, load_diet_labels
from generate_splits import load_node_splits


NUM_CV_FOLDS = 10
MAX_ITER = 1000
FIT_MAX_ITER = 500
N_MODELS = 500
EDG_FILE = "data/edges.tsv"


def _load_sweep_dataset(
    model_num: int,
    p: float,
    q: float,
    gamma: float,
    seed: int,
    n2v_mode: str,
    label_name: str,
) -> Dataset:
    """
    Build a Dataset for one of the 5 per-fold embeddings used to evaluate
    node2vec+ parameters. model_num is the held-out fold: nodes whose split
    equals model_num are the test set, everything else (including nodes
    with no defined split - always train, never held out) is training data.
    """
    emb_file = f"emb_cache/emb_p_{p}_q_{q}_g_{gamma}.tsv"
    emb = load_or_create_embedding(EDG_FILE, emb_file, n2v_mode, p, q, gamma, seed)

    labels = load_time_labels() if label_name == "time" else load_diet_labels()
    splits = load_node_splits()
    joined = labels.join(splits, how="inner")

    train_idx = joined.index[joined["split"] != model_num]
    test_idx = joined.index[joined["split"] == model_num]

    dataset = Dataset.from_tables(
        label_name=label_name,
        feature_table=emb,
        labels=joined.loc[train_idx, "label"],
        cv_folds=NUM_CV_FOLDS,
    )
    dataset.test_data = emb.loc[test_idx]
    dataset.test_labels = joined.loc[test_idx, "label"].to_numpy()
    return dataset


def log_model(classifier_type: str, model: int, params: dict) -> None:
    """
    log model hyperparameters to wandb
    """
    for parameter, value in params.items():
        wandb.summary[f"{classifier_type}_model_{model}_{parameter}"] = value


def log_final_metrics(scores: dict) -> None:
    """
    log aggregate metrics for each classifier
    and an overall embedding score to wandb
    """
    emb_score = 0
    for classifier_type, avg_scores in scores.items():
        classifier_score = sum(avg_scores) / len(avg_scores)
        wandb.summary[f"{classifier_type}_avg_val_f1"] = classifier_score
        emb_score += classifier_score
    wandb.summary["emb_score"] = emb_score / 2


def train_loop(
    p: float, q: float, gamma: int, random_seed: int, n2v_mode: str, save: bool
) -> None:
    """
    trains and evaluates all models associated with an embedding space
    """
    pred_columns = {"time": ["baseline", "endpoint"], "diet": ["dairy", "meat"]}
    scores = {"time": [], "diet": []}
    for model_num in range(1, 6):
        for classifier_type in ["time", "diet"]:
            # load data associated with model
            dataset = _load_sweep_dataset(
                model_num, p, q, gamma, random_seed, n2v_mode, classifier_type
            )
            clf = LogisticRegressionClassifier(
                label_name=classifier_type,
                pred_columns=pred_columns[classifier_type],
                seed=random_seed,
                cv_max_iter=MAX_ITER,
                n_iter_search=N_MODELS,
                fit_max_iter=FIT_MAX_ITER,
            )
            result = clf.run_sweep(dataset, NUM_CV_FOLDS)
            # record per-fold CV scores found through cv
            for fold, score in result.fold_scores.items():
                wandb.summary[f"{classifier_type}_model_{model_num}_val_{fold}_f1"] = (
                    score
                )
            # record parameters found through cv
            log_model(classifier_type, model_num, result.best_params)
            # record avg score for CV
            scores[classifier_type].append(result.best_score)
            # log train/test metrics for the final model trained on all training data
            for split, metrics in [
                ("train", result.train_metrics),
                ("test", result.test_metrics),
            ]:
                for metric, score in metrics.items():
                    wandb.log(
                        {f"{classifier_type}_model_{model_num}_{split}_{metric}": score}
                    )
                    wandb.summary[
                        f"{classifier_type}_model_{model_num}_{split}_{metric}"
                    ] = score
            if save:
                clf.save(
                    f"results/models/{classifier_type}_{model_num}_p_{p}_q_{q}_g_{gamma}.pkl"
                )
    log_final_metrics(scores)
