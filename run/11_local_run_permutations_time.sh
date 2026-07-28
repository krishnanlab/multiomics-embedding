#!/usr/bin/env sh

# Usage: ./run/11_local_run_permutations_time.sh
# Runs 100,000 permutations (100 batches of 1000) as local concurrent
# subprocesses. Run run/10_local_fit_permutation_test_time.sh first.
exec conda run -n multiomics python scripts/run_permutations.py \
    --fitted-state results/permutations/time/fit/fitted_state.pkl \
    --n-permutations 100000 --batch-size 1000 --base-seed 0 --max_jobs 4 \
    --out results/permutations/time/100000_permutations
