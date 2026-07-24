import pytest

from src.deployment import DeploymentRunner, DeploymentTask


def make_task(**overrides):
    kwargs = dict(label_tsv="labels.tsv", pred_columns=["neg", "pos"])
    kwargs.update(overrides)
    return DeploymentTask(**kwargs)


def test_deployment_task_valid_construction():
    make_task()


@pytest.mark.parametrize("bad", [["only_one"], ["a", "b", "c"], ["same", "same"]])
def test_deployment_task_rejects_bad_pred_columns(bad):
    with pytest.raises(ValueError):
        make_task(pred_columns=bad)


def make_runner(**overrides):
    kwargs = dict(
        edg_file="e.tsv",
        split_tsv="splits.tsv",
        feature_paths=["f1.txt"],
        labels={"t": make_task()},
    )
    kwargs.update(overrides)
    return DeploymentRunner(**kwargs)


def test_deployment_runner_valid_construction():
    make_runner()


def test_deployment_runner_rejects_empty_labels():
    with pytest.raises(ValueError, match="labels"):
        make_runner(labels={})


def test_deployment_runner_rejects_empty_feature_paths():
    with pytest.raises(ValueError, match="feature_paths"):
        make_runner(feature_paths=[])
