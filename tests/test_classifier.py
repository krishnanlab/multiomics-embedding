import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit, cross_val_score

from src.classifier import LogisticRegressionClassifier
from src.dataset import Dataset


def make_clf(**overrides):
    kwargs = dict(
        label_name="t",
        pred_columns=["neg", "pos"],
        seed=42,
        cv_max_iter=100,
        n_iter_search=10,
    )
    kwargs.update(overrides)
    return LogisticRegressionClassifier(**kwargs)


def test_valid_construction():
    make_clf()


def test_pred_columns_none_is_ok():
    make_clf(pred_columns=None)


@pytest.mark.parametrize("bad", [["only_one"], ["a", "b", "c"], ["same", "same"]])
def test_rejects_bad_pred_columns(bad):
    with pytest.raises(ValueError):
        make_clf(pred_columns=bad)


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_rejects_bad_cv_max_iter(bad):
    with pytest.raises((ValueError, TypeError)):
        make_clf(cv_max_iter=bad)


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_rejects_bad_n_iter_search(bad):
    with pytest.raises((ValueError, TypeError)):
        make_clf(n_iter_search=bad)


def test_fit_max_iter_none_is_ok():
    make_clf(fit_max_iter=None)


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_rejects_bad_fit_max_iter(bad):
    with pytest.raises((ValueError, TypeError)):
        make_clf(fit_max_iter=bad)


def test_cv_fold_scores_matches_independent_cross_val_score():
    """_cv_fold_scores pulls split{i}_test_{metric} out of cv_results_ by
    hand-formatted key - checked here against cross_val_score, an
    independent sklearn code path, as ground truth."""
    X = pd.DataFrame({"x0": [0, 0, 0, 1, 1, 1], "x1": [0, 1, 0, 1, 0, 1]})
    y = np.array([0, 0, 0, 1, 1, 1])
    cv = PredefinedSplit(np.array([0, 0, 1, 1, 2, 2]))
    dataset = Dataset(label_name="t", train_data=X, train_labels=y, cv_folds=cv)

    clf = make_clf(
        n_iter_search=1,
        param_distributions=[{"C": [1.0], "penalty": ["l2"], "solver": ["lbfgs"]}],
        scoring="f1",
        pred_columns=None,
    )
    clf.cv_search(dataset)
    fold_scores = clf._cv_fold_scores("score", n_folds=3)

    expected = cross_val_score(
        LogisticRegression(random_state=42, max_iter=100, C=1.0, penalty="l2", solver="lbfgs"),
        X, y, cv=cv, scoring="f1",
    )
    assert [fold_scores[i] for i in range(3)] == pytest.approx(list(expected))
