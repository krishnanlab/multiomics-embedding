import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import PredefinedSplit

from src.dataset import (
    Dataset,
    require_positive_int,
    require_unique_index,
    validate_pred_columns,
)


def make_data(n=6):
    train_data = pd.DataFrame({"x": range(n)}, index=[f"n{i}" for i in range(n)])
    train_labels = np.array([0, 1] * (n // 2))
    return train_data, train_labels


# --- require_unique_index ---


def test_require_unique_index_passes_on_unique():
    require_unique_index(pd.Index(["a", "b", "c"]), "thing")


def test_require_unique_index_raises_on_duplicate():
    with pytest.raises(ValueError, match="duplicate"):
        require_unique_index(pd.Index(["a", "b", "a"]), "thing")


# --- validate_pred_columns ---


def test_validate_pred_columns_none_is_ok():
    validate_pred_columns(None)


def test_validate_pred_columns_valid_pair_is_ok():
    validate_pred_columns(["neg", "pos"])


@pytest.mark.parametrize("bad", [["only_one"], ["a", "b", "c"], ["same", "same"]])
def test_validate_pred_columns_raises(bad):
    with pytest.raises(ValueError):
        validate_pred_columns(bad)


# --- require_positive_int ---


def test_require_positive_int_passes():
    require_positive_int(5, "n")


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_require_positive_int_raises(bad):
    with pytest.raises((ValueError, TypeError)):
        require_positive_int(bad, "n")


# --- Dataset.__post_init__ ---


def test_dataset_valid_construction():
    train_data, train_labels = make_data()
    ds = Dataset(label_name="t", train_data=train_data, train_labels=train_labels, cv_folds=5)
    assert ds.label_name == "t"


def test_dataset_accepts_predefined_split_cv_folds():
    train_data, train_labels = make_data()
    cv = PredefinedSplit(np.array([0, 1, 0, 1, 0, 1]))
    Dataset(label_name="t", train_data=train_data, train_labels=train_labels, cv_folds=cv)


def test_dataset_rejects_bad_cv_folds_type():
    train_data, train_labels = make_data()
    with pytest.raises(TypeError):
        Dataset(label_name="t", train_data=train_data, train_labels=train_labels, cv_folds="5")


def test_dataset_rejects_length_mismatch():
    train_data, _ = make_data(6)
    with pytest.raises(ValueError, match="rows"):
        Dataset(
            label_name="t",
            train_data=train_data,
            train_labels=np.array([0, 1, 0]),
            cv_folds=5,
        )


def test_dataset_rejects_nan_labels():
    train_data, _ = make_data()
    labels = np.array([0.0, 1.0, np.nan, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="NaN"):
        Dataset(label_name="t", train_data=train_data, train_labels=labels, cv_folds=5)


def test_dataset_rejects_non_binary_labels():
    train_data, _ = make_data()
    labels = np.array([0, 1, 2, 1, 0, 1])
    with pytest.raises(ValueError, match="binary"):
        Dataset(label_name="t", train_data=train_data, train_labels=labels, cv_folds=5)


def test_dataset_rejects_only_one_of_test_data_test_labels():
    train_data, train_labels = make_data()
    with pytest.raises(ValueError, match="test_data and test_labels"):
        Dataset(
            label_name="t",
            train_data=train_data,
            train_labels=train_labels,
            cv_folds=5,
            test_data=train_data,
        )


def test_dataset_rejects_test_length_mismatch():
    train_data, train_labels = make_data()
    with pytest.raises(ValueError, match="test_data"):
        Dataset(
            label_name="t",
            train_data=train_data,
            train_labels=train_labels,
            cv_folds=5,
            test_data=train_data,
            test_labels=np.array([0, 1]),
        )


def test_dataset_rejects_duplicate_train_data_index():
    train_data = pd.DataFrame({"x": [1, 2, 3]}, index=["a", "a", "b"])
    with pytest.raises(ValueError, match="duplicate"):
        Dataset(
            label_name="t",
            train_data=train_data,
            train_labels=np.array([0, 1, 0]),
            cv_folds=3,
        )


def test_dataset_rejects_duplicate_feature_matrix_index():
    train_data, train_labels = make_data()
    feature_matrix = pd.DataFrame({"x": [1, 2]}, index=["f1", "f1"])
    with pytest.raises(ValueError, match="duplicate"):
        Dataset(
            label_name="t",
            train_data=train_data,
            train_labels=train_labels,
            cv_folds=5,
            feature_matrix=feature_matrix,
        )


# --- Dataset.from_label_tsv join/filter correctness ---


def test_from_label_tsv_drops_sentinel_rows_and_builds_matching_folds(tmp_path):
    label_tsv = tmp_path / "labels.tsv"
    split_tsv = tmp_path / "splits.tsv"
    label_tsv.write_text("node\tlabel\nn0\t0\nn1\t1\nn2\t0\nn3\t1\n")
    split_tsv.write_text("node\tsplit\nn0\t1\nn1\t2\nn2\t-1\nn3\t2\n")
    feature_table = pd.DataFrame({"x": [1, 2, 3, 4]}, index=["n0", "n1", "n2", "n3"])

    dataset = Dataset.from_label_tsv(
        label_name="t",
        feature_table=feature_table,
        label_tsv=str(label_tsv),
        split_tsv=str(split_tsv),
    )

    assert sorted(dataset.train_data.index) == ["n0", "n1", "n3"]
    assert dict(zip(dataset.train_data.index, dataset.train_labels)) == {
        "n0": 0,
        "n1": 1,
        "n3": 1,
    }
    assert isinstance(dataset.cv_folds, PredefinedSplit)
    assert set(dataset.cv_folds.test_fold) == {1, 2}


def test_from_label_tsv_restricts_to_samples_path(tmp_path):
    label_tsv = tmp_path / "labels.tsv"
    split_tsv = tmp_path / "splits.tsv"
    samples_path = tmp_path / "samples.txt"
    label_tsv.write_text("node\tlabel\nn0\t0\nn1\t1\nn2\t0\n")
    split_tsv.write_text("node\tsplit\nn0\t1\nn1\t1\nn2\t1\n")
    samples_path.write_text("n0\nn1\n")
    feature_table = pd.DataFrame({"x": [1, 2, 3]}, index=["n0", "n1", "n2"])

    dataset = Dataset.from_label_tsv(
        label_name="t",
        feature_table=feature_table,
        label_tsv=str(label_tsv),
        split_tsv=str(split_tsv),
        samples_path=str(samples_path),
    )

    assert sorted(dataset.train_data.index) == ["n0", "n1"]


def test_from_label_tsv_feature_matrix_only_contains_feature_paths(tmp_path):
    label_tsv = tmp_path / "labels.tsv"
    split_tsv = tmp_path / "splits.tsv"
    feature_path = tmp_path / "features.txt"
    label_tsv.write_text("node\tlabel\nn0\t0\nn1\t1\n")
    split_tsv.write_text("node\tsplit\nn0\t1\nn1\t1\n")
    feature_path.write_text("f0\nf1\n")
    feature_table = pd.DataFrame(
        {"x": [1, 2, 3, 4]}, index=["n0", "n1", "f0", "f1"]
    )

    dataset = Dataset.from_label_tsv(
        label_name="t",
        feature_table=feature_table,
        label_tsv=str(label_tsv),
        split_tsv=str(split_tsv),
        feature_paths=[str(feature_path)],
    )

    assert sorted(dataset.feature_matrix.index) == ["f0", "f1"]
