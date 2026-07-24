#!/usr/bin/env sh

# Usage: ./run/local_04_initial_sweep_wandb.sh <wandb_username>
# Random-searches node2vec+ params (p, q, gamma), ranked by time-point
# validation F1. Requires `wandb login`. Runs trials as local concurrent
# subprocesses (--max_jobs 4) - see run/slurm_04_initial_sweep_wandb.sh to
# submit each trial as its own SLURM job instead.
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <wandb_username>"
  exit 1
fi

USERNAME="$1"

exec conda run -n multiomics \
     python scripts/run_sweep.py \
       --username "$USERNAME" \
       --runs 100 \
       --max_jobs 4 \
       --name time_point \
       --metric time_avg_val_f1 \
       --p_max 100 \
       --q_max 100 \
       --edges-file data/edges.tsv \
       --samples-file data/nodes/samples.txt \
       --feature-files data/nodes/microbes.txt data/nodes/metabolites.txt
