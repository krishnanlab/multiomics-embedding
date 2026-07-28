#!/usr/bin/env sh

# Usage: ./run/12_local_combine_permutations_diet.sh
# Same as run/12_local_combine_permutations_time.sh, for diet.
set -e
conda run -n multiomics python scripts/slice_permutation_manifest.py \
    --manifest results/permutations/diet/100000_permutations/manifest.json \
    --n-permutations 10000 \
    --out results/permutations/diet/10000_permutations/manifest.json
conda run -n multiomics python scripts/combine_permutations.py \
    --manifest results/permutations/diet/10000_permutations/manifest.json \
    --out results/permutations/diet/10000_permutations/combined.tsv
conda run -n multiomics python scripts/combine_permutations.py \
    --manifest results/permutations/diet/100000_permutations/manifest.json \
    --out results/permutations/diet/100000_permutations/combined.tsv
