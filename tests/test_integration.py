import warnings

from src.deployment import DeploymentRunner, DeploymentTask
from src.sweep import EmbeddingParams


def build_tiny_graph(tmp_path):
    samples = [f"s{i}" for i in range(6)]
    features = ["f0", "f1"]
    labels = [0, 1, 0, 1, 0, 1]
    folds = [1, 1, 1, 2, 2, 2]

    edg_file = tmp_path / "edges.tsv"
    edges = [
        f"{s}\t{f}\t{1.0 + 0.1 * i}"
        for i, s in enumerate(samples)
        for f in features
    ]
    edg_file.write_text("\n".join(edges))

    label_tsv = tmp_path / "labels.tsv"
    label_tsv.write_text(
        "node\tlabel\n" + "\n".join(f"{s}\t{l}" for s, l in zip(samples, labels))
    )
    split_tsv = tmp_path / "splits.tsv"
    split_tsv.write_text(
        "node\tsplit\n" + "\n".join(f"{s}\t{f}" for s, f in zip(samples, folds))
    )
    feature_file = tmp_path / "features.txt"
    feature_file.write_text("\n".join(features))

    return edg_file, label_tsv, split_tsv, feature_file


def test_embed_classify_predict_end_to_end(tmp_path):
    """A tiny synthetic graph run through the real pipeline: node2vec+
    embedding generation (real pecanpy call, not mocked) -> cv_search ->
    fit_full -> predict_features."""
    edg_file, label_tsv, split_tsv, feature_file = build_tiny_graph(tmp_path)

    task = DeploymentTask(label_tsv=str(label_tsv), pred_columns=["neg", "pos"])
    runner = DeploymentRunner(
        edg_file=str(edg_file),
        split_tsv=str(split_tsv),
        feature_paths=[str(feature_file)],
        labels={"t": task},
        scoring="f1",
        refit=True,
        emb_cache_dir=str(tmp_path / "emb_cache"),
        cv_max_iter=50,
        fit_max_iter=50,
        n_iter_search=1,
    )
    params = EmbeddingParams(
        p=1.0, q=1.0, gamma=0.0, dim=4, num_walks=2, walk_length=4,
        window_size=2, workers=1, seed=0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn convergence warnings on tiny data
        results = runner.run(params, save_to=None, log_wandb=False)

    preds = results["t"]["feature_predictions"]
    assert sorted(preds.index) == ["f0", "f1"]
    assert list(preds.columns) == ["neg", "pos"]
    assert preds.to_numpy().min() >= 0
    assert preds.to_numpy().max() <= 1
    assert (preds.sum(axis=1).round(6) == 1.0).all()

    assert set(results["t"]["best_params"]) >= {"C", "penalty", "solver"}
    assert 0 <= results["combined_score"] <= 1
