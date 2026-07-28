#!/usr/bin/env sh

# Usage: ./run/11_local_run_permutations_diet.sh
# Same as run/11_local_run_permutations_time.sh, for diet. Run
# run/10_local_fit_permutation_test_diet.sh first.
exec conda run -n multiomics python scripts/run_permutations.py \
    --fitted-state results/permutations/diet/fit/fitted_state.pkl \
    --n-permutations 100000 --batch-size 1000 --base-seed 0 --max_jobs 4 \
    --out results/permutations/diet/100000_permutations
