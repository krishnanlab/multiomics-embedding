import pandas as pd
import pytest

from src.zscoring import FeatureZScorer


def test_valid_construction():
    FeatureZScorer(feature_lists={"microbes": ["m1", "m2"], "metabolites": ["b1"]})


def test_rejects_empty_dict():
    with pytest.raises(ValueError, match="non-empty"):
        FeatureZScorer(feature_lists={})


def test_rejects_empty_subset_list():
    with pytest.raises(ValueError, match="empty"):
        FeatureZScorer(feature_lists={"microbes": ["m1"], "metabolites": []})


def test_score_computes_zscore_within_subset():
    preds = pd.DataFrame({"pos": [1.0, 2.0, 3.0]}, index=["m1", "m2", "m3"])
    scorer = FeatureZScorer({"microbes": ["m1", "m2", "m3"]})
    scored = scorer.score(preds)

    # mean=2, sample std (ddof=1)=1 -> z = [-1, 0, 1]
    expected = pd.Series([-1.0, 0.0, 1.0], index=["m1", "m2", "m3"], name="pos")
    pd.testing.assert_series_equal(scored["microbes"]["pos"], expected)


def test_score_subsets_are_independent():
    """Each named subset is z-scored against only its own rows, not the whole preds table."""
    preds = pd.DataFrame(
        {"pos": [1.0, 2.0, 3.0, 100.0, 200.0, 300.0]},
        index=["m1", "m2", "m3", "b1", "b2", "b3"],
    )
    scorer = FeatureZScorer({"microbes": ["m1", "m2", "m3"], "metabolites": ["b1", "b2", "b3"]})
    scored = scorer.score(preds)

    # both subsets are the same shape scaled by 100x - identical z-scores
    # here prove the metabolite values didn't leak into the microbe mean/std
    expected = pd.Series([-1.0, 0.0, 1.0])
    assert list(scored["microbes"]["pos"]) == pytest.approx(list(expected))
    assert list(scored["metabolites"]["pos"]) == pytest.approx(list(expected))


def test_score_warns_on_zero_variance_subset():
    preds = pd.DataFrame({"pos": [0.5, 0.5, 0.5]}, index=["m1", "m2", "m3"])
    scorer = FeatureZScorer({"microbes": ["m1", "m2", "m3"]})
    with pytest.warns(UserWarning, match="zero/undefined variance"):
        scored = scorer.score(preds)
    assert scored["microbes"]["pos"].isna().all()


def test_score_warns_on_single_row_subset():
    preds = pd.DataFrame({"pos": [0.5]}, index=["m1"])
    scorer = FeatureZScorer({"microbes": ["m1"]})
    with pytest.warns(UserWarning, match="zero/undefined variance"):
        scorer.score(preds)
