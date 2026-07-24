import warnings

import pandas as pd
import pytest

from src.sweep import EmbeddingParams, SweepRunner, Task


# --- Task ---


def make_labels(values):
    return pd.DataFrame({"label": values}, index=[f"n{i}" for i in range(len(values))])


def test_task_valid_construction():
    Task(labels=make_labels([0, 1, 0, 1]), pred_columns=["neg", "pos"])


def test_task_rejects_bad_pred_columns():
    with pytest.raises(ValueError):
        Task(labels=make_labels([0, 1]), pred_columns=["only_one"])


def test_task_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="binary"):
        Task(labels=make_labels([0, 1, 2]), pred_columns=["neg", "pos"])


def test_task_rejects_nan_labels():
    with pytest.raises(ValueError, match="NaN"):
        Task(labels=make_labels([0, 1, None]), pred_columns=["neg", "pos"])


# --- EmbeddingParams ---


def test_embedding_params_valid_construction():
    EmbeddingParams(p=1.0, q=1.0, gamma=0.0)


def test_embedding_params_rejects_bad_n2v_mode():
    with pytest.raises(ValueError, match="n2v_mode"):
        EmbeddingParams(p=1.0, q=1.0, gamma=0.0, n2v_mode="bogus")


@pytest.mark.parametrize("field", ["dim", "num_walks", "walk_length", "window_size", "workers"])
def test_embedding_params_rejects_non_positive_ints(field):
    with pytest.raises((ValueError, TypeError)):
        EmbeddingParams(p=1.0, q=1.0, gamma=0.0, **{field: 0})


# --- SweepRunner ---


def make_node_splits(splits):
    return pd.DataFrame({"split": splits}, index=[f"n{i}" for i in range(len(splits))])


def make_task_labels():
    return {
        "t": Task(labels=make_labels([0, 1, 0, 1, 0, 1]), pred_columns=["neg", "pos"])
    }


def test_sweep_runner_valid_construction():
    node_splits = make_node_splits([1, 2, -1, 1, 2, -1])
    runner = SweepRunner(edg_file="e.tsv", node_splits=node_splits, labels=make_task_labels())
    assert runner.num_outer_folds == 2


def test_sweep_runner_rejects_missing_split_column():
    node_splits = pd.DataFrame({"not_split": [1, 2]})
    with pytest.raises(ValueError, match="split"):
        SweepRunner(edg_file="e.tsv", node_splits=node_splits, labels=make_task_labels())


def test_sweep_runner_rejects_empty_labels():
    node_splits = make_node_splits([1, 2, -1])
    with pytest.raises(ValueError, match="labels"):
        SweepRunner(edg_file="e.tsv", node_splits=node_splits, labels={})


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_sweep_runner_rejects_bad_inner_cv_folds(bad):
    node_splits = make_node_splits([1, 2, -1])
    with pytest.raises((ValueError, TypeError)):
        SweepRunner(
            edg_file="e.tsv",
            node_splits=node_splits,
            labels=make_task_labels(),
            inner_cv_folds=bad,
        )


def test_sweep_runner_rejects_non_integer_split_values():
    node_splits = make_node_splits([1.5, 2, -1])
    with pytest.raises(ValueError, match="non-integer"):
        SweepRunner(edg_file="e.tsv", node_splits=node_splits, labels=make_task_labels())


def test_sweep_runner_rejects_non_positive_split_values_after_sentinel_filter():
    # 0 is not the sentinel (-1) here, so it must be treated as a real,
    # invalid (non-positive) fold value - this is exactly the two-convention
    # divergence the split-sentinel fix closes
    node_splits = make_node_splits([1, 0, -1])
    with pytest.raises(ValueError, match="non-positive"):
        SweepRunner(edg_file="e.tsv", node_splits=node_splits, labels=make_task_labels())


def test_sweep_runner_respects_custom_sentinel():
    # with sentinel=0, the 0 rows are excluded (not treated as a fold)
    node_splits = make_node_splits([1, 2, 0, 0])
    runner = SweepRunner(
        edg_file="e.tsv",
        node_splits=node_splits,
        labels=make_task_labels(),
        no_split_sentinel=0,
    )
    assert runner.num_outer_folds == 2


def test_sweep_runner_warns_on_fold_gap():
    # folds {1, 2, 4} - fold 3 missing
    node_splits = make_node_splits([1, 2, 4, -1])
    with pytest.warns(UserWarning, match="missing outer fold"):
        SweepRunner(edg_file="e.tsv", node_splits=node_splits, labels=make_task_labels())


def test_sweep_runner_no_warning_on_contiguous_folds():
    node_splits = make_node_splits([1, 2, 3, -1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        SweepRunner(edg_file="e.tsv", node_splits=node_splits, labels=make_task_labels())


def test_sweep_runner_sentinel_matches_dataset_from_label_tsv_convention(tmp_path):
    """
    Regression test for the split-sentinel bug: Dataset.from_label_tsv's
    no_split_sentinel filter and SweepRunner's fold-counting must agree on
    which rows are "never held out," for any sentinel value, not just -1.
    """
    from src.dataset import Dataset

    label_tsv = tmp_path / "labels.tsv"
    split_tsv = tmp_path / "splits.tsv"
    label_tsv.write_text("node\tlabel\nn0\t0\nn1\t1\nn2\t0\nn3\t1\n")
    # use 0 as the sentinel instead of -1
    split_tsv.write_text("node\tsplit\nn0\t1\nn1\t2\nn2\t0\nn3\t0\n")
    feature_table = pd.DataFrame({"x": [1, 2, 3, 4]}, index=["n0", "n1", "n2", "n3"])

    dataset = Dataset.from_label_tsv(
        label_name="t",
        feature_table=feature_table,
        label_tsv=str(label_tsv),
        split_tsv=str(split_tsv),
        no_split_sentinel=0,
    )
    # from_label_tsv drops the sentinel rows (n2, n3) -> 2 rows remain
    assert len(dataset.train_labels) == 2

    node_splits = pd.read_csv(split_tsv, sep="\t", index_col="node")
    runner = SweepRunner(
        edg_file="e.tsv",
        node_splits=node_splits,
        labels=make_task_labels(),
        no_split_sentinel=0,
    )
    # same two rows (folds 1, 2) are the only real folds under this sentinel
    assert runner.num_outer_folds == 2


# --- SweepRunner._dataset_for_fold ---


def test_dataset_for_fold_splits_train_test_by_fold():
    labels = make_labels([0, 1, 0, 1])
    node_splits = make_node_splits([1, 1, 2, 2])
    runner = SweepRunner(
        edg_file="e.tsv",
        node_splits=node_splits,
        labels={"t": Task(labels=labels, pred_columns=["neg", "pos"])},
    )
    emb = pd.DataFrame({"x": [10, 20, 30, 40]}, index=labels.index)

    dataset = runner._dataset_for_fold("t", fold=1, emb=emb)

    assert sorted(dataset.test_data.index) == ["n0", "n1"]
    assert sorted(dataset.train_data.index) == ["n2", "n3"]
    assert list(dataset.test_labels) == [0, 1]
    assert list(dataset.train_labels) == [0, 1]
