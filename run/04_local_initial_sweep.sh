#!/usr/bin/env sh

# Usage: ./run/04_local_initial_sweep.sh
# Non-wandb counterpart to run/04_local_initial_sweep_wandb.sh - same
# random search over node2vec+ params, no wandb account needed. Results
# land as local JSONs under --out instead of being logged to wandb. Runs
# trials as local concurrent subprocesses (--max_jobs 4).
exec conda run -n multiomics \
     python scripts/run_sweep_local.py \
       --runs 100 \
       --max_jobs 4 \
       --out results/initial_sweep \
       --p_max 100 \
       --q_max 100 \
       --edges-file data/edges.tsv \
       --samples-file data/nodes/samples.txt \
       --feature-files data/nodes/microbes.txt data/nodes/metabolites.txt
