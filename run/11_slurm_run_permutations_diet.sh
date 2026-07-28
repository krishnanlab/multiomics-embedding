#!/usr/bin/env sh

# Usage: ./run/11_slurm_run_permutations_diet.sh
# Same as run/11_slurm_run_permutations_time.sh, for diet.
exec conda run -n multiomics python scripts/run_permutations.py \
    --fitted-state results/permutations/diet/fit/fitted_state.pkl \
    --n-permutations 100000 --batch-size 1000 --base-seed 0 \
    --out results/permutations/diet/100000_permutations \
    --slurm --slurm-time 00:20:00 --slurm-mem 7GB --slurm-cpus 2
