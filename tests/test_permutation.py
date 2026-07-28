import numpy as np
import pandas as pd
import pytest

from src.permutation import CONSENSUS_MODES, PermutationTest, _validate_best_params, combine
from src.zscoring import FeatureZScorer


# --- _validate_best_params (direct unit tests) ---


def test_validate_best_params_accepts_valid_dict():
    _validate_best_params({"C": 1.0, "penalty": "l1", "solver": "liblinear"})


def test_validate_best_params_raises_on_unknown_key():
    with pytest.raises(ValueError, match="unknown"):
        _validate_best_params({"C": 1.0, "penalty": "l1", "solver": "liblinear", "bogus": 1})


def test_validate_best_params_warns_on_missing_expected_key():
    with pytest.warns(UserWarning, match="missing expected"):
        _validate_best_params({"C": 1.0, "penalty": "l1"})  # no "solver"


def test_validate_best_params_no_warning_for_legitimately_sparse_dict():
    import warnings

    # C/penalty/solver present, no l1_ratio - that's fine, l1_ratio is
    # elasticnet-only and not in the "always expected" set
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _validate_best_params({"C": 1.0, "penalty": "l2", "solver": "lbfgs"})


# --- PermutationTest.__init__ (checks that fail before any file I/O) ---


def make_zscorer():
    return FeatureZScorer({"microbes": ["m1"]})


def make_kwargs(**overrides):
    kwargs = dict(
        embeddings=[pd.DataFrame({"x": [1]})],
        label_name="t",
        label_tsv="nonexistent_labels.tsv",
        split_tsv="nonexistent_splits.tsv",
        zscorer=make_zscorer(),
        best_params=[{"C": 1.0, "penalty": "l1", "solver": "liblinear"}],
    )
    kwargs.update(overrides)
    return kwargs


def test_rejects_empty_embeddings():
    with pytest.raises(ValueError, match="embeddings"):
        PermutationTest(**make_kwargs(embeddings=[], best_params=[]))


def test_rejects_best_params_length_mismatch():
    with pytest.raises(ValueError, match="best_params"):
        PermutationTest(**make_kwargs(best_params=[]))


def test_rejects_non_positive_threshold():
    with pytest.raises(ValueError, match="threshold"):
        PermutationTest(**make_kwargs(threshold=0))


def test_rejects_prob_threshold_below_half():
    with pytest.raises(ValueError, match="prob_threshold"):
        PermutationTest(**make_kwargs(prob_threshold=0.49))


@pytest.mark.parametrize("bad", [["only_one"], ["a", "b", "c"], ["same", "same"]])
def test_rejects_bad_pred_columns(bad):
    with pytest.raises(ValueError):
        PermutationTest(**make_kwargs(pred_columns=bad))


def test_rejects_best_params_with_unknown_key():
    with pytest.raises(ValueError, match="unknown"):
        PermutationTest(**make_kwargs(best_params=[{"C": 1.0, "bogus": 1}]))


# --- combine() correctness (hand-rolled stats, not a library wrapper -
# this is where the earlier per-group-vs-combined BH bug lived) ---


def test_combine_p_value_formula():
    observed = pd.Series({"f1": 0.8, "f2": 0.2, "f3": 1.0})
    null = pd.DataFrame(
        {
            0: [0.9, 0.1, 0.5],
            1: [0.5, 0.3, 0.6],
            2: [0.7, 0.1, 0.4],
            3: [0.6, 0.9, 0.3],
        },
        index=["f1", "f2", "f3"],
    )
    result = combine(observed, pd.Series({"f1": 1, "f2": 1, "f3": 1}), null)

    # f1: null >= 0.8 -> just 0.9 -> 1 exceedance -> p = (1+1)/(1+4) = 0.4
    assert result.loc["f1", "p_value"] == pytest.approx(0.4)
    # f2: null >= 0.2 -> 0.3, 0.9 -> 2 exceedances -> p = (1+2)/(1+4) = 0.6
    assert result.loc["f2", "p_value"] == pytest.approx(0.6)
    # f3: null >= 1.0 -> none -> 0 exceedances -> p = (1+0)/(1+4) = 0.2 (the floor)
    assert result.loc["f3", "p_value"] == pytest.approx(0.2)

    assert result["n_permutations"].eq(4).all()
    assert not result.loc["f1", "at_permutation_floor"]
    assert not result.loc["f2", "at_permutation_floor"]
    assert result.loc["f3", "at_permutation_floor"]


def test_combine_warns_on_feature_not_covered_by_any_group():
    features = ["f1", "f2"]
    observed = pd.Series({"f1": 0.8, "f2": 0.2})
    direction = pd.Series(1, index=features)
    null = pd.DataFrame({0: [0.1, 0.1], 1: [0.2, 0.2]}, index=features)

    with pytest.warns(UserWarning, match="not covered"):
        result = combine(observed, direction, null, feature_groups={"a": ["f1"]})
    assert pd.isna(result.loc["f2", "q_value"])
    assert not pd.isna(result.loc["f1", "q_value"])


def test_combine_single_group_matches_ungrouped():
    """A single feature_groups entry containing every feature must give
    identical q-values to omitting feature_groups entirely - grouping
    everything together is the same as not grouping."""
    rng = np.random.default_rng(0)
    features = [f"f{i}" for i in range(10)]
    observed = pd.Series(rng.random(10), index=features)
    null = pd.DataFrame(rng.random((10, 20)), index=features)
    direction = pd.Series(1, index=features)

    ungrouped = combine(observed, direction, null)
    grouped = combine(observed, direction, null, feature_groups={"all": features})

    pd.testing.assert_series_equal(
        ungrouped["q_value"], grouped["q_value"], check_names=False
    )


def test_combine_per_group_bh_differs_from_combined():
    """Two groups with very different p-value distributions must be
    corrected independently (this is the actual behavior the BH bug broke:
    it pooled every feature into one correction regardless of group)."""
    # group A: 2 features with strong signal (small p-values)
    # group B: 20 features with no signal (p-values near 1)
    n_permutations = 100
    rng = np.random.default_rng(1)

    a_features = ["a0", "a1"]
    b_features = [f"b{i}" for i in range(20)]
    observed = pd.Series(
        {**{f: 0.99 for f in a_features}, **{f: 0.5 for f in b_features}}
    )
    direction = pd.Series(1, index=observed.index)
    a_null = pd.DataFrame(  # rarely reaches observed -> small p-values
        rng.uniform(0, 0.5, size=(len(a_features), n_permutations)), index=a_features
    )
    b_null = pd.DataFrame(  # centered at observed -> p-values near 0.5
        rng.uniform(0.3, 0.7, size=(len(b_features), n_permutations)), index=b_features
    )
    null = pd.concat([a_null, b_null])

    combined = combine(observed, direction, null)
    grouped = combine(
        observed, direction, null, feature_groups={"a": a_features, "b": b_features}
    )

    # pooling group A's 2 strong features in with group B's 20 null features
    # penalizes them under combined BH; correcting group A alone doesn't
    assert (grouped.loc[a_features, "q_value"] < combined.loc[a_features, "q_value"]).all()


# --- run_trial()'s within-fold permutation invariant ---


def make_permutation_test(tmp_path, **overrides):
    samples = [f"s{i}" for i in range(6)]
    features = ["f0", "f1"]
    folds = [1, 1, 1, 2, 2, 2]
    labels = [0, 1, 0, 1, 0, 1]

    label_tsv = tmp_path / "labels.tsv"
    split_tsv = tmp_path / "splits.tsv"
    feature_file = tmp_path / "features.txt"
    label_tsv.write_text(
        "node\tlabel\n" + "\n".join(f"{s}\t{l}" for s, l in zip(samples, labels))
    )
    split_tsv.write_text(
        "node\tsplit\n" + "\n".join(f"{s}\t{f}" for s, f in zip(samples, folds))
    )
    feature_file.write_text("\n".join(features))

    rng = np.random.default_rng(0)
    all_nodes = samples + features
    emb = pd.DataFrame(rng.normal(size=(len(all_nodes), 2)), index=all_nodes)

    kwargs = dict(
        embeddings=[emb],
        label_name="t",
        label_tsv=str(label_tsv),
        split_tsv=str(split_tsv),
        zscorer=FeatureZScorer({"features": features}),
        best_params=[{"C": 1.0, "penalty": "l2", "solver": "lbfgs"}],
        pred_columns=["neg", "pos"],
        feature_paths=[str(feature_file)],
        seed=0,
        fit_max_iter=200,
    )
    kwargs.update(overrides)
    return PermutationTest(**kwargs)


def test_run_trial_permutes_within_fold_only(tmp_path):
    test = make_permutation_test(tmp_path)
    original_labels = test.datasets[0].train_labels.copy()

    rng = np.random.default_rng(42)
    permuted = original_labels.copy()
    for idx in test._fold_groups.values():
        permuted[idx] = rng.permutation(permuted[idx])

    # every fold's own set of labels (pos/neg counts) must be unchanged -
    # only their assignment within the fold may move
    for idx in test._fold_groups.values():
        assert sorted(permuted[idx]) == sorted(original_labels[idx])
    assert sorted(permuted) == sorted(original_labels)


def test_save_load_round_trip(tmp_path):
    import warnings

    test = make_permutation_test(tmp_path)
    save_path = tmp_path / "state.pkl"
    test.save(str(save_path))
    loaded = PermutationTest.load(str(save_path))

    assert loaded.label_name == test.label_name
    assert loaded.best_params == test.best_params
    assert loaded.threshold == test.threshold
    assert loaded.prob_threshold == test.prob_threshold
    np.testing.assert_array_equal(
        loaded.datasets[0].train_labels, test.datasets[0].train_labels
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original_trial = test.run_trial(np.random.default_rng(0))
        loaded_trial = loaded.run_trial(np.random.default_rng(0))
    pd.testing.assert_frame_equal(original_trial, loaded_trial)


def test_run_trial_and_observed_run_end_to_end(tmp_path):
    import warnings

    test = make_permutation_test(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn convergence warnings on tiny data
        observed = test.observed()
        trial = test.run_trial(np.random.default_rng(0))

    assert set(observed.scores.index) == {"f0", "f1"}
    assert set(observed.scores.columns) == set(CONSENSUS_MODES)
    assert observed.scores["hit_fraction_z"].between(0, 1).all()
    assert observed.scores["mean_prob"].between(0, 1).all()
    assert observed.scores["median_prob"].between(0, 1).all()
    assert observed.scores["hit_fraction_prob"].between(0, 1).all()

    assert set(trial.index) == {"f0", "f1"}
    assert set(trial.columns) == set(CONSENSUS_MODES)
    assert trial["hit_fraction_z"].between(0, 1).all()
