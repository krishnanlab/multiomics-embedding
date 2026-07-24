"""
Author: Keenan Manpearl
Date: 2026-07-22

Generic node2vec+ -> classifier deployment runner. Given an edge list,
node2vec+ params, and any number of binary-classification label tsvs:
generates/loads the embedding, then fits a final classifier per label on
*all* available data, optionally saving model/weights/feature-predictions.

- labels: dict[str, DeploymentTask], one entry per classifier - each names
  its own label tsv/pred columns (see DeploymentTask). samples_path is
  per-task; feature_paths/split_tsv are shared across labels.
"""

from dataclasses import dataclass

import pandas as pd
import wandb

from src.dataset import Dataset, validate_pred_columns
from src.sweep import BaseRunner, EmbeddingParams


@dataclass
class DeploymentTask:
    """
    One binary classifier to train as part of a deployment run.
    samples_path is per-task since which samples need a label varies by
    label (e.g. diet only applies to endpoint samples - samples_path=None
    trusts label_tsv's own rows instead of validating against a list).
    """

    label_tsv: str  # path: two columns, node/label (0/1)
    pred_columns: list[str]  # [negative_class_name, positive_class_name]
    samples_path: str | None = None

    def __post_init__(self) -> None:
        validate_pred_columns(self.pred_columns)


class DeploymentRunner(BaseRunner):
    """
    Fits a final classifier per label on all available data for one
    embedding space (no held-out test set - see module docstring).
    """

    def __init__(
        self,
        edg_file: str,
        split_tsv: str,
        feature_paths: list[str],
        labels: dict[str, DeploymentTask],
        scoring: str | list[str] = "f1",
        refit: bool | str = True,
        **kwargs,
    ) -> None:
        super().__init__(edg_file, **kwargs)
        if not labels:
            raise ValueError("labels must be non-empty")
        if not feature_paths:
            raise ValueError("feature_paths must be non-empty")
        self.split_tsv = split_tsv
        self.feature_paths = feature_paths
        self.labels = labels
        self.scoring = scoring
        self.refit = refit

    def _dataset_for_label(self, label_name: str, emb: pd.DataFrame) -> Dataset:
        task = self.labels[label_name]
        return Dataset.from_label_tsv(
            label_name=label_name,
            feature_table=emb,
            label_tsv=task.label_tsv,
            split_tsv=self.split_tsv,
            samples_path=task.samples_path,
            feature_paths=self.feature_paths,
        )

    def run(
        self,
        params: EmbeddingParams,
        save_to: str | None = None,
        embedding: pd.DataFrame | None = None,
        log_wandb: bool = False,
    ) -> dict:
        """
        Generate/load the embedding (or use the one given), then fit a
        final model per label on all data. Returns each label's best
        hyperparameters, feature predictions, and median/IQR/mean CV score
        (from the deployment fit's own search - no held-out fold here, so
        this is the closest equivalent to SweepRunner's val score), plus
        an overall combined_score.
        """
        emb = self._load_embedding(params, embedding)
        if save_to:
            self._ensure_save_dir(save_to)
        # RandomizedSearchCV names the score column "score" for a single
        # scoring string, or the metric's own name (must be in scoring)
        # for a list of scoring metrics - same convention as
        # LogisticRegressionClassifier.run_sweep's score_metric
        score_metric = self.refit if isinstance(self.scoring, list) else "score"

        results: dict = {}
        scores: dict[str, list[float]] = {}
        for label_name, task in self.labels.items():
            dataset = self._dataset_for_label(label_name, emb)
            clf = self._make_classifier(
                label_name,
                task.pred_columns,
                params.seed,
                scoring=self.scoring,
                refit=self.refit,
                n_jobs=params.workers,
            )
            result = clf.run_deployment(dataset)
            n_folds = dataset.cv_folds.get_n_splits()
            fold_scores = clf._cv_fold_scores(score_metric, n_folds)
            scores[label_name] = list(fold_scores.values())
            if log_wandb:
                self._log_label_wandb(label_name, fold_scores, result.best_params)
            if save_to:
                tag = params.cache_tag(self.edg_file)
                if isinstance(self.scoring, list):
                    with open(f"{save_to}/{label_name}_{tag}_logging.txt", "w") as f:
                        f.write("============== Node2Vec Parameters ==============\n")
                        for param_name, value in vars(params).items():
                            f.write(f"{param_name}: {value}\n")
                        f.write("\n")
                        clf.write_results(f, n_folds=n_folds)
                clf.save(f"{save_to}/{label_name}_{tag}_model.pkl")
                clf.save_weights(f"{save_to}/{label_name}_{tag}_weights.txt")
                if result.feature_predictions is not None:
                    result.feature_predictions.to_csv(
                        f"{save_to}/{label_name}_{tag}_feature_predictions.tsv",
                        sep="\t",
                    )
            results[label_name] = {
                "best_params": result.best_params,
                "feature_predictions": result.feature_predictions,
            }

        metrics = self._aggregate(scores)
        if log_wandb:
            self._log_summary_wandb(metrics)
        for label_name in results:
            results[label_name].update(metrics[label_name])

        full_results = {"edg_file": self.edg_file, **vars(params), **results,
                         "combined_score": metrics["combined_score"]}
        if save_to:
            self._save_results_json(save_to, params, full_results)
        return full_results

    @staticmethod
    def _log_label_wandb(label_name: str, fold_scores: dict, best_params: dict) -> None:
        for fold_idx, score in fold_scores.items():
            wandb.summary[f"{label_name}_val_{fold_idx}_f1"] = score
        for param, value in best_params.items():
            wandb.summary[f"{label_name}_{param}"] = value
