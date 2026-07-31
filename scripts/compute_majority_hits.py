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


def compute_hit_counts(deploy_dir: str) -> pd.DataFrame:
    """Wide table: index=feature, one <label>_<direction> column per entry in
    LABEL_DIRECTIONS holding the number of deployment embeddings where that
    direction's probability was > 0.5, plus an n_embeddings column."""
    columns = {}
    n_embeddings = None
    for label, directions in LABEL_DIRECTIONS.items():
        paths = sorted(glob.glob(f"{deploy_dir}/*/{label}_feature_predictions.tsv"))
        if not paths:
            raise FileNotFoundError(f"no */{label}_feature_predictions.tsv files found under {deploy_dir}")
        if n_embeddings is None:
            n_embeddings = len(paths)
        elif len(paths) != n_embeddings:
            raise ValueError(
                f"{label} has {len(paths)} embeddings, expected {n_embeddings} (mismatch vs. other label)"
            )
        for direction in directions:
            probs = load_per_embedding_predictions(paths, column=direction)
            columns[f"{label}_{direction}"] = (probs > 0.5).sum(axis=1)

    table = pd.DataFrame(columns)
    table.index.name = "feature"
    table["n_embeddings"] = n_embeddings
    return table


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

    table = compute_hit_counts(args.deploy_dir)
    hit_cols = [c for c in table.columns if c != "n_embeddings"]

    intermediate_path = out_dir / "num_hits.tsv"
    table.to_csv(intermediate_path, sep="\t")
    print(f"{len(table)} features x {len(hit_cols)} directions -> {intermediate_path}")

    for min_hits in sorted({args.majority_min_hits, args.unanimous_min_hits}):
        any_hit = (table[hit_cols] >= min_hits).any(axis=1)
        subset = table[any_hit].assign(max_hits=table.loc[any_hit, hit_cols].max(axis=1))
        subset = subset.sort_values("max_hits", ascending=False)
        out_path = out_dir / f"majority_hits_{min_hits}.tsv"
        subset.to_csv(out_path, sep="\t")
        print(f"{len(subset)}/{len(table)} features with >={min_hits} hits in any direction -> {out_path}")
