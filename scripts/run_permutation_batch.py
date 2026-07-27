"""
Author: Keenan Manpearl
Date: 2026-07-24

Worker script for scripts/run_permutations.py: loads a fitted
PermutationTest (from scripts/fit_permutation_test.py) and runs one
batch of permutation trials, saving the per-feature null scores for
EVERY CONSENSUS_MODES mode (see src/permutation.py) as a single .npz
file, one array per mode - so scripts/combine_permutations.py can later
compute p-values for any mode without recomputing anything here. One
process = one batch - this is what gets invoked N times, locally or via
SLURM, to parallelize a large permutation count.

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from argparse import ArgumentParser

import numpy as np

from src.permutation import PermutationTest

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fitted-state", required=True)
    parser.add_argument("--n-permutations", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--out", required=True, help=".npz file to write this batch's per-mode null scores to"
    )
    args = parser.parse_args()

    test = PermutationTest.load(args.fitted_state)
    batch = test.run_batch(args.n_permutations, args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **{mode: df.to_numpy(dtype=np.float32) for mode, df in batch.items()})
    n_trials = next(iter(batch.values())).shape[1]
    n_features = next(iter(batch.values())).shape[0]
    print(
        f"wrote {n_trials} permutations x {n_features} features x {len(batch)} modes "
        f"to {args.out}"
    )
