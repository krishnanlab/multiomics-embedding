#!/usr/bin/env sh

# Usage: ./run/04_slurm_initial_sweep.sh
# Same as run/04_local_initial_sweep.sh, but submits each trial as
# its own SLURM job instead of running it as a local subprocess. No wandb
# account needed. Use a QOS that allows multiple concurrent jobs per user.
exec conda run -n multiomics \
     python scripts/run_sweep_local.py \
       --runs 100 \
       --out results/initial_sweep \
       --p_max 100 \
       --q_max 100 \
       --edges-file data/edges.tsv \
       --samples-file data/nodes/samples.txt \
       --feature-files data/nodes/microbes.txt data/nodes/metabolites.txt \
       --slurm
