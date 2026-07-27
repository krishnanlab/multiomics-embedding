"""
Author: Keenan Manpearl
Date: 2026-07-24

Derives a smaller permutation-testing tier from a larger one's manifest.json
without recomputing anything: takes the first N batches (in batch_id order)
whose permutation counts sum exactly to --n-permutations, and writes a new
manifest referencing those same batch .npy files. Pairs with
scripts/run_permutations.py's --extend (see its module docstring) - e.g. a
1,000-permutation tier is just the first 10 batches of a 100-batch/
10,000-permutation run, if every batch is 100 permutations.

    python scripts/slice_permutation_manifest.py \\
        --manifest results/permutations_time_10000/manifest.json \\
        --n-permutations 1000 \\
        --out results/permutations_time_1000/manifest.json

The sliced manifest works directly with scripts/combine_permutations.py -
no other change needed.

"""

import json
import os
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--n-permutations", required=True, type=int)
    parser.add_argument("--out", required=True, help="path to write the sliced manifest.json to")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    batches = sorted(manifest["batches"], key=lambda b: b["batch_id"])
    selected = []
    total = 0
    for batch in batches:
        if total >= args.n_permutations:
            break
        selected.append(batch)
        total += batch["n_permutations"]

    if total != args.n_permutations:
        raise ValueError(
            f"batches don't evenly divide to exactly {args.n_permutations} "
            f"permutations - got {total} from the first {len(selected)} batch(es). "
            "This requires the source manifest's batches to all be the same "
            "size and --n-permutations to be an exact multiple of it."
        )

    sliced = dict(manifest)
    sliced["n_permutations"] = args.n_permutations
    sliced["n_batches"] = len(selected)
    sliced["batches"] = selected
    sliced["sliced_from"] = args.manifest

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(sliced, f, indent=2)
    print(f"wrote {len(selected)} batch(es) ({total} permutations) to {args.out}")
