#!/usr/bin/env sh

# Usage: ./run/12_local_combine_permutations_time.sh
# Derives the 10,000-permutation tier as the first 10 batches of the
# 100,000-permutation run (no recomputation - see
# scripts/slice_permutation_manifest.py), then combines both tiers into
# final p-value/q-value tables, one file per mode. Run
# run/{local,slurm}_11_run_permutations_time.sh first.
set -e
conda run -n multiomics python scripts/slice_permutation_manifest.py \
    --manifest results/permutations/time/100000_permutations/manifest.json \
    --n-permutations 10000 \
    --out results/permutations/time/10000_permutations/manifest.json
conda run -n multiomics python scripts/combine_permutations.py \
    --manifest results/permutations/time/10000_permutations/manifest.json \
    --out results/permutations/time/10000_permutations/combined.tsv
conda run -n multiomics python scripts/combine_permutations.py \
    --manifest results/permutations/time/100000_permutations/manifest.json \
    --out results/permutations/time/100000_permutations/combined.tsv
