"""
Author: Keenan Manpearl
Date: 2026-07-31

For every feature and every diet/time direction, counts how many of the
deployment embeddings' models
(results/deployment_for_permutations/<tag>/<label>_feature_predictions.tsv,
one <tag> subdirectory per curated embedding - see
scripts/train_deployment_models.py) called that direction with probability
> 0.5. Writes the full per-direction hit-count table plus two threshold
reports: features with a hit count >=--majority-min-hits (a plain majority,
e.g. 4/7) or >=--unanimous-min-hits (unanimous, e.g. 7/7) in any direction.
Each threshold report also carries the per-embedding raw probability (one
column per embedding tag) and the cross-embedding average probability, both
for the direction/class each feature was predicted for (predicted_class).
"""

import glob
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.da_comparison import load_per_embedding_predictions

LABEL_DIRECTIONS = {
    "diet": ["dairy", "meat"],
    "time": ["baseline", "endpoint"],
}


def compute_hit_counts(deploy_dir: str) -> tuple[pd.DataFrame, dict, list]:
    """Wide table: index=feature, one <label>_<direction> column per entry in
    LABEL_DIRECTIONS holding the number of deployment embeddings where that
    direction's probability was > 0.5, plus an n_embeddings column. Also
    returns probs_by_col (the same <label>_<direction> keys mapped to their
    (n_features, n_embeddings) raw-probability DataFrame, columns named by
    embedding tag) and the list of embedding tags, in column order."""
    columns = {}
    probs_by_col = {}
    tags = None
    n_embeddings = None
    for label, directions in LABEL_DIRECTIONS.items():
        paths = sorted(glob.glob(f"{deploy_dir}/*/{label}_feature_predictions.tsv"))
        if not paths:
            raise FileNotFoundError(f"no */{label}_feature_predictions.tsv files found under {deploy_dir}")
        if n_embeddings is None:
            n_embeddings = len(paths)
            tags = [Path(p).parent.name for p in paths]
        elif len(paths) != n_embeddings:
            raise ValueError(
                f"{label} has {len(paths)} embeddings, expected {n_embeddings} (mismatch vs. other label)"
            )
        for direction in directions:
            probs = load_per_embedding_predictions(paths, column=direction)
            probs.columns = tags
            probs_by_col[f"{label}_{direction}"] = probs
            columns[f"{label}_{direction}"] = (probs > 0.5).sum(axis=1)

    table = pd.DataFrame(columns)
    table.index.name = "feature"
    table["n_embeddings"] = n_embeddings
    return table, probs_by_col, tags


def predicted_class_probs(predicted_col: pd.Series, probs_by_col: dict, tags: list) -> pd.DataFrame:
    """For each feature (row of predicted_col, valued with the <label>_<direction>
    column it was predicted for), pull that direction's raw per-embedding
    probabilities from probs_by_col. Returns a (len(predicted_col), n_tags)
    DataFrame of raw probabilities, columns named by embedding tag."""
    result = pd.DataFrame(index=predicted_col.index, columns=tags, dtype=float)
    for col_name, probs in probs_by_col.items():
        rows = predicted_col.index[predicted_col == col_name]
        result.loc[rows] = probs.loc[rows]
    return result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--deploy-dir",
        default="results/deployment_for_permutations",
        help="directory containing <tag>/<label>_feature_predictions.tsv subdirectories (default: results/deployment_for_permutations)",
    )
    parser.add_argument(
        "--out-dir",
        default="results/deployment_for_permutations",
        help="where to write num_hits.tsv and majority_hits_<n>.tsv (default: results/deployment_for_permutations)",
    )
    parser.add_argument("--majority-min-hits", type=int, default=4, help="plain-majority hit threshold (default: 4)")
    parser.add_argument("--unanimous-min-hits", type=int, default=7, help="unanimous hit threshold (default: 7)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table, probs_by_col, tags = compute_hit_counts(args.deploy_dir)
    hit_cols = [c for c in table.columns if c != "n_embeddings"]

    intermediate_path = out_dir / "num_hits.tsv"
    table.to_csv(intermediate_path, sep="\t")
    print(f"{len(table)} features x {len(hit_cols)} directions -> {intermediate_path}")

    for min_hits in sorted({args.majority_min_hits, args.unanimous_min_hits}):
        any_hit = (table[hit_cols] >= min_hits).any(axis=1)
        subset = table[any_hit].assign(max_hits=table.loc[any_hit, hit_cols].max(axis=1))
        predicted_class = table.loc[any_hit, hit_cols].idxmax(axis=1)
        class_probs = predicted_class_probs(predicted_class, probs_by_col, tags)
        subset = subset.assign(predicted_class=predicted_class, avg_probability=class_probs.mean(axis=1))
        subset = pd.concat([subset, class_probs], axis=1)
        subset = subset.sort_values("max_hits", ascending=False)
        out_path = out_dir / f"majority_hits_{min_hits}.tsv"
        subset.to_csv(out_path, sep="\t")
        print(f"{len(subset)}/{len(table)} features with >={min_hits} hits in any direction -> {out_path}")
