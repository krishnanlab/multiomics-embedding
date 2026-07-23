"""
Trains and evaluates a logistic regression classifier against a Dataset.

One class is reused for every binary label a caller trains against (e.g.
"diet", "time point", ...) and every pipeline that needs a classifier -
label_name and pred_columns are just tags/config supplied by the caller,
nothing about a particular study's labels is hardcoded here.

Required data format
---------------------
See src/dataset.py's Dataset docstring for what train_data/train_labels/
cv_folds/feature_matrix must look like. pred_columns, if given, must be a
2-element list naming the two classes in the order
LogisticRegression.predict_proba returns them (i.e.
[negative_class_name, positive_class_name] for labels encoded as 0/1) -
if omitted, feature predictions are left labeled with the raw 0/1 classes.

"""

import pickle
from dataclasses import dataclass
from typing import TextIO

import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.dataset import Dataset


def iqr(scores: list[float]) -> float:
    """interquartile range (75th percentile - 25th percentile)"""
    return np.percentile(scores, 75) - np.percentile(scores, 25)


@dataclass
class SweepResult:
    """Everything a caller needs to report on one run_sweep() call."""

    best_params: dict
    best_score: float
    fold_scores: dict[int, float]
    train_metrics: dict
    test_metrics: dict


@dataclass
class DeploymentResult:
    """Everything a caller needs to report on one run_deployment() call."""

    best_params: dict
    feature_predictions: pd.DataFrame | None


DEFAULT_PARAM_DISTRIBUTIONS = [
    {"C": uniform(0.1, 500.0), "penalty": ["l1"], "solver": ["liblinear"]},
    {"C": uniform(0.1, 500.0), "penalty": ["l2"], "solver": ["lbfgs"]},
    {
        "C": uniform(0.1, 500.0),
        "penalty": ["elasticnet"],
        "l1_ratio": uniform(0, 1),
        "solver": ["saga"],
    },
]


class LogisticRegressionClassifier:
    """
    Trains and evaluates a logistic regression classifier for one binary label.
    """

    def __init__(
        self,
        label_name: str,
        pred_columns: list[str] | None = None,
        *,
        seed: int,
        cv_max_iter: int,
        n_iter_search: int,
        fit_max_iter: int | None = None,
        scoring: str | list[str] = "f1",
        refit: bool | str = True,
        param_distributions: list[dict] | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.label_name = label_name
        self.pred_columns = pred_columns
        self.seed = seed
        self.cv_max_iter = cv_max_iter
        self.n_iter_search = n_iter_search
        self.fit_max_iter = fit_max_iter
        self.scoring = scoring
        self.refit = refit
        self.param_distributions = param_distributions or DEFAULT_PARAM_DISTRIBUTIONS
        self.n_jobs = n_jobs
        self.search_: RandomizedSearchCV | None = None
        self.model_: LogisticRegression | None = None

    def cv_search(self, dataset: Dataset) -> "LogisticRegressionClassifier":
        """
        use a RandomizedSearchCV to find the best hyperparameters,
        validating on dataset.cv_folds. n_jobs parallelizes across
        candidate x fold combinations via joblib - None (the sklearn
        default) means single-threaded, -1 means "all available cores".
        """
        log_reg = LogisticRegression(random_state=self.seed, max_iter=self.cv_max_iter)
        clf = RandomizedSearchCV(
            log_reg,
            self.param_distributions,
            n_iter=self.n_iter_search,
            cv=dataset.cv_folds,
            scoring=self.scoring,
            refit=self.refit,
            random_state=self.seed,
            n_jobs=self.n_jobs,
        )
        self.search_ = clf.fit(dataset.train_data, dataset.train_labels)
        return self

    def fit_full(
        self, dataset: Dataset, params: dict | None = None
    ) -> "LogisticRegressionClassifier":
        """
        train a logistic regression model on all of dataset.train_data,
        using params (defaults to the best params found by cv_search)
        """
        params = params if params is not None else self.search_.best_params_
        max_iter = (
            self.fit_max_iter if self.fit_max_iter is not None else self.cv_max_iter
        )
        model = LogisticRegression(random_state=self.seed, max_iter=max_iter)
        model.set_params(**params)
        model.fit(dataset.train_data, dataset.train_labels)
        self.model_ = model
        return self

    def evaluate(self, dataset: Dataset, split: str) -> dict:
        """
        compute accuracy/balanced_accuracy/precision/recall/f1 for the given split
        ("train" or "test" - "test" requires dataset.test_data/test_labels to be set)
        """
        if split == "train":
            data, labels = dataset.train_data, dataset.train_labels
        else:
            data, labels = dataset.test_data, dataset.test_labels
        predictions = self.model_.predict(data)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "balanced_accuracy": balanced_accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions),
            "recall": recall_score(labels, predictions),
            "f1": f1_score(labels, predictions),
        }

    def _cv_fold_scores(self, metric: str, n_folds: int) -> dict[int, float]:
        """
        extract the per-fold validation score for the given metric from the
        best-performing hyperparameter combination found by cv_search
        """
        results = self.search_.cv_results_
        best_idx = self.search_.best_index_
        return {i: results[f"split{i}_test_{metric}"][best_idx] for i in range(n_folds)}

    def write_results(self, file: TextIO, n_folds: int) -> None:
        """
        write the best CV hyperparameters and per-fold validation scores
        (median/IQR/variance) for each metric in self.scoring to an
        already-open file. Requires self.scoring to be a list of metric names.
        """
        for param, value in self.search_.best_params_.items():
            file.write(f"best {param}: {value}\n")
        file.write("\n")
        for metric in self.scoring:
            scores = self._cv_fold_scores(metric, n_folds)
            file.write(f"{metric} validation scores\n")
            for fold, s in scores.items():
                file.write(f"fold {fold}: {s}\n")
            file.write(f"median score: {np.median(list(scores.values()))}\n")
            file.write(f"IQR: {iqr(list(scores.values()))}\n")
            file.write(f"variance: {np.var(list(scores.values()))}\n")
            file.write("\n")

    def predict_features(self, dataset: Dataset) -> pd.DataFrame | None:
        """
        generate predictions for each feature in dataset.features_to_predict(),
        or None if this dataset doesn't support feature-level predictions.
        Columns are named by pred_columns ([negative_class_name,
        positive_class_name]) if given, else left as the raw 0/1 labels
        (self.model_.classes_).
        """
        features = dataset.features_to_predict()
        if features is None:
            return None
        probs = self.model_.predict_proba(features)
        columns = self.pred_columns if self.pred_columns is not None else self.model_.classes_
        return pd.DataFrame(probs, index=features.index, columns=columns)

    def run_sweep(
        self, dataset: Dataset, n_folds: int, score_metric: str = "score"
    ) -> SweepResult:
        """
        Run the standard hyperparameter-sweep-evaluation procedure: search for
        the best hyperparameters via cv_search, train a final model on all
        training data, and evaluate it on the train split and (if present)
        the test split. Returns the results for the caller to report however
        it likes (e.g. to wandb) - score_metric is the cv_results_ column
        name to pull per-fold scores from (RandomizedSearchCV names it
        "score" for a single scoring string, or the metric's own name for a
        list of scoring metrics).
        """
        self.cv_search(dataset)
        fold_scores = self._cv_fold_scores(score_metric, n_folds)
        self.fit_full(dataset)
        train_metrics = self.evaluate(dataset, "train")
        test_metrics = (
            self.evaluate(dataset, "test") if dataset.test_data is not None else {}
        )
        return SweepResult(
            best_params=self.search_.best_params_,
            best_score=self.search_.best_score_,
            fold_scores=fold_scores,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
        )

    def run_deployment(self, dataset: Dataset) -> DeploymentResult:
        """
        Run the standard deployment procedure: search for the best
        hyperparameters via cv_search, train a final model on all training
        data, and (if the dataset supports it) predict on its feature
        matrix. Call write_results/save/save_weights afterward to persist
        output - this method only returns results, it doesn't write anything.
        """
        self.cv_search(dataset)
        self.fit_full(dataset)
        return DeploymentResult(
            best_params=self.search_.best_params_,
            feature_predictions=self.predict_features(dataset),
        )

    def save(self, path: str) -> None:
        """save the trained model to path"""
        with open(path, "wb") as f:
            pickle.dump(self.model_, f)

    def save_weights(self, path: str) -> None:
        """save the trained model's coefficients to path"""
        coefficients = self.model_.coef_[0]
        with open(path, "w") as f:
            f.writelines(f"{coef}\n" for coef in coefficients)
