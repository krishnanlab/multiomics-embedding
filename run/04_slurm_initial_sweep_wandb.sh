#!/usr/bin/env sh

# Usage: ./run/04_slurm_initial_sweep_wandb.sh <wandb_username>
# Same as run/04_local_initial_sweep_wandb.sh, but submits each of the 100
# trials as its own SLURM job (see jobs/template.sh, scripts/slurm_utils.py)
# instead of running them as local subprocesses. Requires `wandb login` and
# a working jobs/template.sh. Use a QOS that allows multiple concurrent
# jobs per user - a 1-job-at-a-time QOS defeats the point of --slurm.
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <wandb_username>"
  exit 1
fi

USERNAME="$1"

exec conda run -n multiomics \
     python scripts/run_sweep.py \
       --username "$USERNAME" \
       --runs 100 \
       --name time_point \
       --metric time_avg_val_f1 \
       --p_max 100 \
       --q_max 100 \
       --edges-file data/edges.tsv \
       --samples-file data/nodes/samples.txt \
       --feature-files data/nodes/microbes.txt data/nodes/metabolites.txt \
       --slurm
