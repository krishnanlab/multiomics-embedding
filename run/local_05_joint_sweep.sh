#!/usr/bin/env sh

# Usage: ./run/local_05_joint_sweep.sh
# Non-wandb counterpart to run/local_05_joint_sweep_wandb.sh - same random
# search, no wandb account needed. Results land as local JSONs under --out.
# Runs trials as local concurrent subprocesses (--max_jobs 4).
exec conda run -n multiomics \
     python scripts/run_sweep_local.py \
       --runs 200 \
       --max_jobs 4 \
       --out results/joint_sweep \
       --edges-file data/edges.tsv \
       --samples-file data/nodes/samples.txt \
       --feature-files data/nodes/microbes.txt data/nodes/metabolites.txt
