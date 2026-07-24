"""
Author: Keenan Manpearl
Date: 2026-07-22

Generic node2vec+ -> classifier hyperparameter-sweep runner. Given an edge
list, node2vec+ params, and any number of binary-classification label sets:
generates/loads the embedding, builds one Dataset per outer-CV-fold x label,
runs LogisticRegressionClassifier.run_sweep on each, and aggregates
median/IQR/mean validation F1 across outer folds - optionally to wandb.

BaseRunner holds what this shares with src/deployment.py's DeploymentRunner
(embedding generation/caching, classifier construction) - see that module
for the deployment-mode "fit on samples, predict on features" counterpart
to this sweep-mode "nested CV, held-out fold" runner.

- labels: dict[str, Task], one entry per classifier to train. See Task.
- node_splits: DataFrame indexed by node, integer "split" column - which
  outer fold a node is held out in, or non-positive if never held out.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

from src.classifier import LogisticRegressionClassifier, iqr
from src.dataset import Dataset
from src.embedding import load_or_create_embedding


@dataclass
class Task:
    """One binary classifier to train as part of a sweep run."""

    labels: pd.DataFrame  # index=node, single "label" column (0/1)
    pred_columns: list[str]  # [negative_class_name, positive_class_name]


@dataclass
class EmbeddingParams:
    """node2vec+ parameters identifying one embedding space."""

    p: float
    q: float
    gamma: float
    dim: int = 128
    num_walks: int = 10
    walk_length: int = 80
    window_size: int = 10
    n2v_mode: str = "OTF"
    seed: int = 42
    workers: int = 4  # not part of cache_tag - affects speed, not the embedding itself

    def cache_tag(self, edg_file: str) -> str:
        """a filename-safe tag identifying this embedding space"""
        edg_tag = Path(edg_file).stem
        return (
            f"{edg_tag}_p_{self.p}_q_{self.q}_g_{self.gamma}"
            f"_dim_{self.dim}_nw_{self.num_walks}_wl_{self.walk_length}"
            f"_ws_{self.window_size}"
        )


class BaseRunner:
    """
    Shared plumbing for SweepRunner and DeploymentRunner: embedding
    generation/loading/caching and classifier construction. Subclasses
    define what to *do* with an embedding (nested-CV sweep vs.
    fit-on-everything deployment).
    """

    def __init__(
        self,
        edg_file: str,
        emb_cache_dir: str = "emb_cache",
        cv_max_iter: int = 1000,
        fit_max_iter: int = 500,
        n_iter_search: int = 500,
    ) -> None:
        self.edg_file = edg_file
        self.emb_cache_dir = emb_cache_dir
        self.cv_max_iter = cv_max_iter
        self.fit_max_iter = fit_max_iter
        self.n_iter_search = n_iter_search

    @staticmethod
    def _ensure_save_dir(save_to: str) -> None:
        """create save_to (and any missing parents) before writing any file into it"""
        os.makedirs(save_to, exist_ok=True)

    def _save_results_json(self, save_to: str, params: EmbeddingParams, results: dict) -> None:
        """Persist a run's results dict (metrics, not the model) as JSON. DataFrame values (e.g. feature_predictions, saved separately) become a short note instead of being dumped inline."""

        def _sanitize(value):
            if isinstance(value, pd.DataFrame):
                return f"<DataFrame {value.shape[0]}x{value.shape[1]}, saved separately>"
            if isinstance(value, dict):
                return {k: _sanitize(v) for k, v in value.items()}
            return value

        path = f"{save_to}/results_{params.cache_tag(self.edg_file)}.json"
        with open(path, "w") as f:
            json.dump(_sanitize(results), f, indent=2, default=str)

    def _load_embedding(
        self, params: EmbeddingParams, embedding: "pd.DataFrame | None"
    ) -> pd.DataFrame:
        """use the given embedding as-is if provided, else generate/load-from-cache"""
        if embedding is not None:
            return embedding
        emb_file = f"{self.emb_cache_dir}/emb_{params.cache_tag(self.edg_file)}.tsv"
        return load_or_create_embedding(
            self.edg_file,
            emb_file,
            params.n2v_mode,
            params.p,
            params.q,
            params.gamma,
            params.seed,
            params.dim,
            params.num_walks,
            params.walk_length,
            params.window_size,
            params.workers,
        )

    def _make_classifier(
        self,
        label_name: str,
        pred_columns: list[str],
        seed: int,
        **kwargs,
    ) -> LogisticRegressionClassifier:
        return LogisticRegressionClassifier(
            label_name=label_name,
            pred_columns=pred_columns,
            seed=seed,
            cv_max_iter=self.cv_max_iter,
            n_iter_search=self.n_iter_search,
            fit_max_iter=self.fit_max_iter,
            **kwargs,
        )

    @staticmethod
    def _aggregate(scores: dict[str, list[float]]) -> dict:
        """median/IQR/mean val score per label, plus combined_score (mean of means)."""
        metrics: dict = {}
        combined = 0.0
        for label_name, fold_scores in scores.items():
            avg = sum(fold_scores) / len(fold_scores)
            metrics[label_name] = {
                "avg_val_f1": avg,
                "median_val_f1": float(np.median(fold_scores)),
                "iqr_val_f1": float(iqr(fold_scores)),
            }
            combined += avg
        metrics["combined_score"] = combined / len(scores)
        return metrics

    @staticmethod
    def _log_summary_wandb(metrics: dict) -> None:
        for label_name, label_metrics in metrics.items():
            if label_name == "combined_score":
                continue
            for name, value in label_metrics.items():
                wandb.summary[f"{label_name}_{name}"] = value
        wandb.summary["emb_score"] = metrics["combined_score"]


class SweepRunner(BaseRunner):
    """
    Runs the standard embed-then-classify sweep-evaluation procedure for
    one embedding space, against any number of binary classifiers.
    """

    def __init__(
        self,
        edg_file: str,
        node_splits: pd.DataFrame,
        labels: dict[str, Task],
        num_outer_folds: int | None = None,
        inner_cv_folds: int = 10,
        **kwargs,
    ) -> None:
        super().__init__(edg_file, **kwargs)
        self.node_splits = node_splits
        self.labels = labels
        self.num_outer_folds = num_outer_folds or int(
            node_splits.loc[node_splits["split"] > 0, "split"].max()
        )
        self.inner_cv_folds = inner_cv_folds

    def _dataset_for_fold(self, label_name: str, fold: int, emb: pd.DataFrame) -> Dataset:
        """Dataset for one (label, outer fold) pair: split==fold is test, everything else is training."""
        task = self.labels[label_name]
        joined = task.labels.join(self.node_splits, how="inner")
        train_idx = joined.index[joined["split"] != fold]
        test_idx = joined.index[joined["split"] == fold]

        dataset = Dataset.from_tables(
            label_name=label_name,
            feature_table=emb,
            labels=joined.loc[train_idx, "label"],
            cv_folds=self.inner_cv_folds,
        )
        dataset.test_data = emb.loc[test_idx]
        dataset.test_labels = joined.loc[test_idx, "label"].to_numpy()
        return dataset

    def run(
        self,
        params: EmbeddingParams,
        log_wandb: bool = False,
        save_models_to: str | None = None,
        embedding: pd.DataFrame | None = None,
    ) -> dict:
        """
        Generate/load the embedding (or use the one given), then for each
        outer fold x label run a nested-CV classifier search. Always
        returns the median/IQR/mean-f1 + combined_score results dict,
        regardless of log_wandb.
        """
        emb = self._load_embedding(params, embedding)
        if save_models_to:
            self._ensure_save_dir(save_models_to)

        scores: dict[str, list[float]] = {name: [] for name in self.labels}
        for fold in range(1, self.num_outer_folds + 1):
            for label_name, task in self.labels.items():
                dataset = self._dataset_for_fold(label_name, fold, emb)
                clf = self._make_classifier(
                    label_name, task.pred_columns, params.seed, n_jobs=params.workers
                )
                result = clf.run_sweep(dataset, self.inner_cv_folds)
                if log_wandb:
                    self._log_fold_wandb(label_name, fold, result)
                scores[label_name].append(result.best_score)
                if save_models_to:
                    clf.save(
                        f"{save_models_to}/{label_name}_{fold}_"
                        f"{params.cache_tag(self.edg_file)}.pkl"
                    )

        metrics = self._aggregate(scores)
        if log_wandb:
            self._log_summary_wandb(metrics)
        results = {"edg_file": self.edg_file, **vars(params), **metrics}
        if save_models_to:
            self._save_results_json(save_models_to, params, results)
        return results

    @staticmethod
    def _log_fold_wandb(label_name: str, fold: int, result) -> None:
        for fold_idx, score in result.fold_scores.items():
            wandb.summary[f"{label_name}_model_{fold}_val_{fold_idx}_f1"] = score
        for param, value in result.best_params.items():
            wandb.summary[f"{label_name}_model_{fold}_{param}"] = value
        for split, split_metrics in [
            ("train", result.train_metrics),
            ("test", result.test_metrics),
        ]:
            for metric, score in split_metrics.items():
                wandb.log({f"{label_name}_model_{fold}_{split}_{metric}": score})
                wandb.summary[f"{label_name}_model_{fold}_{split}_{metric}"] = score
