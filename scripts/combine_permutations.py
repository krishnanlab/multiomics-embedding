"""
Author: Keenan Manpearl
Date: 2026-07-24

Run once every batch in a run_permutations.py manifest has finished:
concatenates each batch's null scores and combines them with the observed
scores into final p-values/q-values (see src/permutation.py's combine()),
per feature group (microbes vs metabolites), not all features combined.
Reports any missing batch output rather than silently understating
n_permutations - the p-value denominator must be exact.

"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from argparse import ArgumentParser

import numpy as np
import pandas as pd

from src.permutation import combine, PermutationTest

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True, help="tsv to write final p-values/q-values to")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    observed = pd.read_csv(manifest["observed_tsv"], sep="\t", index_col=0)
    feature_groups = PermutationTest.load(manifest["fitted_state"]).zscorer.feature_lists

    null_arrays = []
    missing = []
    total_permutations = 0
    for batch in manifest["batches"]:
        path = Path(batch["npy_path"])
        if not path.exists():
            missing.append(batch)
            continue
        null_arrays.append(np.load(path))
        total_permutations += batch["n_permutations"]

    if missing:
        print(f"{len(missing)} batch(es) have no output yet (still running or failed):")
        for batch in missing:
            print(f"  batch {batch['batch_id']} -> {batch['npy_path']}")
        print()

    if total_permutations != manifest["n_permutations"]:
        print(
            f"WARNING: only {total_permutations} of {manifest['n_permutations']} "
            "requested permutations are present - p-values below are computed "
            "against the smaller count, not the originally requested one."
        )

    null_matrix = pd.DataFrame(np.concatenate(null_arrays, axis=1), index=observed.index)
    result = combine(
        observed["consensus_score"], observed["direction"], null_matrix, feature_groups
    )
    result.to_csv(args.out, sep="\t")
    print(f"wrote {len(result)} features' p-values/q-values to {args.out}")
