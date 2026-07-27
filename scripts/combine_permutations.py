"""
Author: Keenan Manpearl
Date: 2026-07-24

Run once every batch in a run_permutations.py manifest has finished:
concatenates each batch's null scores and combines them with the observed
scores into final p-values/q-values (see src/permutation.py's combine()),
per feature group (microbes vs metabolites), not all features combined.
Reports any missing batch output rather than silently understating
n_permutations - the p-value denominator must be exact.

Every batch's .npz already holds every CONSENSUS_MODES mode (see
src/permutation.py and scripts/run_permutation_batch.py), so switching
which mode(s) you want p-values for is just re-reading the same files -
no permutation trial is ever recomputed just to try a different mode.
--mode picks which ones to compute here (default: every mode present in
observed_tsv); the output is long-format, one row per (feature, mode).

Processes one mode at a time (rereads each batch's .npz once per mode,
pulling out just that mode's array) rather than loading every mode's
null scores into memory at once - for a large run (100k permutations
across 25900 features x 7 modes, float32) holding all 7 simultaneously
is ~70GB+, which OOM-kills on anything but a very large-memory node.
Per-mode is ~7x less peak memory at the cost of rereading each .npz
file multiple times (cheap: reading one named array out of an
uncompressed .npz doesn't touch the others).

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
    parser.add_argument(
        "--mode",
        required=False,
        nargs="+",
        default=None,
        help="CONSENSUS_MODES mode(s) to compute p-values/q-values for "
        "(default: every mode present in observed_tsv)",
    )
    parser.add_argument("--out", required=True, help="tsv to write final p-values/q-values to")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    observed = pd.read_csv(manifest["observed_tsv"], sep="\t", index_col=0)
    feature_groups = PermutationTest.load(manifest["fitted_state"]).zscorer.feature_lists

    modes = args.mode if args.mode is not None else [c for c in observed.columns if c != "direction"]
    unknown_modes = set(modes) - set(observed.columns)
    if unknown_modes:
        parser.error(
            f"--mode {sorted(unknown_modes)} not found in {manifest['observed_tsv']}'s "
            f"columns: {list(observed.columns)}"
        )

    present_batches = []
    missing = []
    total_permutations = 0
    for batch in manifest["batches"]:
        if Path(batch["batch_path"]).exists():
            present_batches.append(batch)
            total_permutations += batch["n_permutations"]
        else:
            missing.append(batch)

    if missing:
        print(f"{len(missing)} batch(es) have no output yet (still running or failed):")
        for batch in missing:
            print(f"  batch {batch['batch_id']} -> {batch['batch_path']}")
        print()

    if total_permutations != manifest["n_permutations"]:
        print(
            f"WARNING: only {total_permutations} of {manifest['n_permutations']} "
            "requested permutations are present - p-values below are computed "
            "against the smaller count, not the originally requested one."
        )

    results = []
    for mode in modes:
        null_arrays = []
        for batch in present_batches:
            with np.load(batch["batch_path"]) as npz:
                null_arrays.append(npz[mode])
        null_matrix = pd.DataFrame(np.concatenate(null_arrays, axis=1), index=observed.index)
        del null_arrays

        result = combine(observed[mode], observed["direction"], null_matrix, feature_groups)
        result.insert(0, "mode", mode)
        results.append(result)
        del null_matrix
    combined = pd.concat(results)
    combined.to_csv(args.out, sep="\t")
    print(f"wrote {len(combined)} rows ({len(modes)} mode(s) x {len(observed)} features) to {args.out}")
