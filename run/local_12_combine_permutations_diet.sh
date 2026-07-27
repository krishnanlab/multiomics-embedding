#!/usr/bin/env sh

# Usage: ./run/local_12_combine_permutations_diet.sh
# Same as run/local_12_combine_permutations_time.sh, for diet.
set -e
conda run -n multiomics python scripts/slice_permutation_manifest.py \
    --manifest results/permutations_diet_10000/manifest.json \
    --n-permutations 1000 \
    --out results/permutations_diet_1000/manifest.json
conda run -n multiomics python scripts/combine_permutations.py \
    --manifest results/permutations_diet_1000/manifest.json \
    --out results/permutations_diet_1000/combined.tsv
conda run -n multiomics python scripts/combine_permutations.py \
    --manifest results/permutations_diet_10000/manifest.json \
    --out results/permutations_diet_10000/combined.tsv
