#!/usr/bin/env sh

# Usage: ./run/11_slurm_run_permutations_time.sh
# Same as run/11_local_run_permutations_time.sh, but submits each of the
# 100 batches as its own SLURM job (real parallelism). --slurm-time/
# --slurm-mem are sized from sacct measurements of actual completed
# permutation batch jobs (~0.5-0.73s/trial, ~0.73GB fixed + ~2.75MB/trial
# peak RSS at 7 modes - see the slurm-job-sizing skill), scaled up for
# CONSENSUS_MODES now having 9 modes.
exec conda run -n multiomics python scripts/run_permutations.py \
    --fitted-state results/permutations/time/fit/fitted_state.pkl \
    --n-permutations 100000 --batch-size 1000 --base-seed 0 \
    --out results/permutations/time/100000_permutations \
    --slurm --slurm-time 00:20:00 --slurm-mem 7GB --slurm-cpus 2
