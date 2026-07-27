#!/usr/bin/env sh

# Usage: ./run/local_12_combine_permutations_time.sh
# Derives the 1,000-permutation tier as the first 10 batches of the
# 10,000-permutation run (no recomputation - see
# scripts/slice_permutation_manifest.py), then combines both tiers into
# final p-value/q-value tables. Run run/{local,slurm}_11_run_permutations_time.sh
# first.
set -e
conda run -n multiomics python scripts/slice_permutation_manifest.py \
    --manifest results/permutations_time_10000/manifest.json \
    --n-permutations 1000 \
    --out results/permutations_time_1000/manifest.json
conda run -n multiomics python scripts/combine_permutations.py \
    --manifest results/permutations_time_1000/manifest.json \
    --out results/permutations_time_1000/combined.tsv
conda run -n multiomics python scripts/combine_permutations.py \
    --manifest results/permutations_time_10000/manifest.json \
    --out results/permutations_time_10000/combined.tsv
