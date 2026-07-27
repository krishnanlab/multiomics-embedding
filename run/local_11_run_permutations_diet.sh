#!/usr/bin/env sh

# Usage: ./run/local_11_run_permutations_diet.sh
# Same as run/local_11_run_permutations_time.sh, for diet. Run
# run/local_10_fit_permutation_test_diet.sh first.
exec conda run -n multiomics python scripts/run_permutations.py \
    --fitted-state results/permutations_diet_fit/fitted_state.pkl \
    --n-permutations 10000 --batch-size 1000 --base-seed 0 --max_jobs 4 \
    --out results/permutations_diet_10000
