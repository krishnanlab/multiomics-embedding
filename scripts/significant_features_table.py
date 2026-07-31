"""
Author: Keenan Manpearl
Date: 2026-07-31

Parses the 100k-permutation combined_mean_confidence.tsv (see
scripts/combine_permutations.py) for diet and time and writes one table per
label of the features whose p_value came out significant - a plain
"what came out of the 100k run" report, not a comparison against the
baseline DA tables (see scripts/da_comparison.py for that).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from argparse import ArgumentParser

from scripts.da_comparison import load_consensus_results, select_features

MODE = "mean_confidence"
LABELS = ("diet", "time")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base-dir",
        default="results/permutations",
        help="directory containing <label>/<n-permutations>_permutations/ (default: results/permutations)",
    )
    parser.add_argument("--n-permutations", type=int, default=100000)
    parser.add_argument("--alpha", type=float, default=0.05, help="p_value significance threshold")
    parser.add_argument(
        "--out-dir",
        default="results/permutations",
        help="where to write <label>_significant_mean_confidence.tsv (default: results/permutations)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label in LABELS:
        combined_path = Path(args.base_dir) / label / f"{args.n_permutations}_permutations" / f"combined_{MODE}.tsv"
        results = load_consensus_results(str(combined_path))

        significant_ids = select_features(results, "p_value", "<", args.alpha)
        table = results.loc[sorted(significant_ids)].sort_values("p_value")

        out_path = out_dir / f"{label}_significant_{MODE}.tsv"
        table.to_csv(out_path, sep="\t")
        print(f"{label}: {len(table)}/{len(results)} features significant at p < {args.alpha} -> {out_path}")
