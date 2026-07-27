"""
Author: Keenan Manpearl
Date: 2026-07-24

One-time setup for permutation testing (see src/permutation.py): builds a
PermutationTest from each embedding's already-known hyperparameters (JSON
via --best-params, or parsed from a deployment run's ..._logging.txt via
--model-logs - no search here), computes the observed consensus score, and
pickles the result for scripts/run_permutations.py's workers.

Run once per label (e.g. diet, time) before scripts/run_permutations.py.

"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from argparse import ArgumentParser

import pandas as pd

from src.permutation import PermutationTest
from src.zscoring import FeatureZScorer


def parse_feature_groups(pairs: list[str]) -> dict[str, str]:
    """parse ["microbes=data/nodes/microbes.txt", ...] into {"microbes": "data/nodes/microbes.txt"}"""
    groups = {}
    for pair in pairs:
        name, _, path = pair.partition("=")
        groups[name] = path
    return groups


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--embeddings", required=True, nargs="+", help="embedding tsv/tsv.gz files"
    )
    best_params_group = parser.add_mutually_exclusive_group(required=True)
    best_params_group.add_argument(
        "--model-logs",
        nargs="+",
        help="one ..._logging.txt path per --embeddings entry (same order), to parse "
        "each embedding's hyperparameters from (see DeploymentRunner's --save-to output)",
    )
    best_params_group.add_argument(
        "--best-params",
        nargs="+",
        help="one JSON hyperparameter dict per --embeddings entry (same order), e.g. "
        '\'{"C": 3.8, "penalty": "l1", "solver": "liblinear"}\'',
    )
    parser.add_argument("--label-name", required=True)
    parser.add_argument("--label-tsv", required=True)
    parser.add_argument("--split-tsv", required=True)
    parser.add_argument("--samples-file", required=False, default=None)
    parser.add_argument(
        "--feature-group",
        required=True,
        nargs="+",
        help="one or more name=path pairs, e.g. microbes=data/nodes/microbes.txt "
        "metabolites=data/nodes/metabolites.txt - used both as Dataset's "
        "feature_paths and as FeatureZScorer's per-subset grouping",
    )
    parser.add_argument(
        "--pred-columns", required=False, nargs=2, default=None, metavar=("NEG", "POS")
    )
    parser.add_argument(
        "--reference-class",
        required=False,
        default=None,
        help="which class's z-score to test (default: pred_columns[-1], or 1 if "
        "--pred-columns isn't given)",
    )
    parser.add_argument("--threshold", required=False, type=float, default=2.0)
    parser.add_argument(
        "--prob-threshold",
        required=False,
        type=float,
        default=0.5,
        help="predict_proba(reference_class) cutoff for the n_confident mode (default: 0.5)",
    )
    parser.add_argument("--seed", required=False, type=int, default=42)
    parser.add_argument("--fit-max-iter", required=False, type=int, default=1000)
    parser.add_argument(
        "--out", required=True, help="directory to write fitted_state.pkl and observed.tsv"
    )
    args = parser.parse_args()

    if args.model_logs is not None and len(args.model_logs) != len(args.embeddings):
        parser.error("--model-logs must have exactly one entry per --embeddings")
    if args.best_params is not None and len(args.best_params) != len(args.embeddings):
        parser.error("--best-params must have exactly one entry per --embeddings")

    best_params = (
        args.model_logs if args.model_logs is not None else [json.loads(p) for p in args.best_params]
    )

    feature_groups = parse_feature_groups(args.feature_group)
    zscorer = FeatureZScorer.from_files(feature_groups)
    embeddings = [pd.read_csv(f, sep="\t", index_col=0) for f in args.embeddings]

    reference_class = args.reference_class
    if reference_class is not None and args.pred_columns is None:
        reference_class = int(reference_class)

    test = PermutationTest(
        embeddings=embeddings,
        label_name=args.label_name,
        label_tsv=args.label_tsv,
        split_tsv=args.split_tsv,
        zscorer=zscorer,
        best_params=best_params,
        pred_columns=args.pred_columns,
        reference_class=reference_class,
        threshold=args.threshold,
        prob_threshold=args.prob_threshold,
        samples_path=args.samples_file,
        feature_paths=list(feature_groups.values()),
        seed=args.seed,
        fit_max_iter=args.fit_max_iter,
    )
    result = test.observed()

    os.makedirs(args.out, exist_ok=True)
    test.save(f"{args.out}/fitted_state.pkl")
    observed_df = result.scores.copy()
    observed_df["direction"] = result.direction
    observed_df.to_csv(f"{args.out}/observed.tsv", sep="\t")
    print(f"setup complete: {len(args.embeddings)} embeddings, {len(observed_df)} features")
    print(f"wrote {args.out}/fitted_state.pkl and {args.out}/observed.tsv")
