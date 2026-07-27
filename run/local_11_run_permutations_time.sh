#!/usr/bin/env sh

# Usage: ./run/local_11_run_permutations_time.sh
# Runs 10,000 permutations (10 batches of 1000) as local concurrent
# subprocesses. Run run/local_10_fit_permutation_test_time.sh first.
#
# To later extend this to 100,000 without recomputing these 10,000: rerun
# with --n-permutations 100000 --batch-size 1000 (same --base-seed) and
# --extend results/permutations_time_10000/manifest.json - see
# scripts/run_permutations.py's module docstring.
exec conda run -n multiomics python scripts/run_permutations.py \
    --fitted-state results/permutations_time_fit/fitted_state.pkl \
    --n-permutations 10000 --batch-size 1000 --base-seed 0 --max_jobs 4 \
    --out results/permutations_time_10000
