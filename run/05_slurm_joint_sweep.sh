#!/usr/bin/env sh

# Usage: ./run/05_slurm_joint_sweep.sh
# Same as run/05_local_joint_sweep.sh, but submits each trial as its
# own SLURM job instead of running it as a local subprocess. No wandb
# account needed. Use a QOS that allows multiple concurrent jobs per user.
exec conda run -n multiomics \
     python scripts/run_sweep_local.py \
       --runs 200 \
       --out results/joint_sweep \
       --edges-file data/edges.tsv \
       --samples-file data/nodes/samples.txt \
       --feature-files data/nodes/microbes.txt data/nodes/metabolites.txt \
       --slurm
