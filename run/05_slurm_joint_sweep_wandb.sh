#!/usr/bin/env sh

# Usage: ./run/05_slurm_joint_sweep_wandb.sh <wandb_username>
# Same as run/05_local_joint_sweep_wandb.sh, but submits each of the 200
# trials as its own SLURM job instead of running them as local subprocesses.
# Requires `wandb login` and a working jobs/template.sh. Use a QOS that
# allows multiple concurrent jobs per user.
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <wandb_username>"
  exit 1
fi

USERNAME="$1"

exec conda run -n multiomics \
     python scripts/run_sweep.py \
       --username "$USERNAME" \
       --runs 200 \
       --name joint_optimization_test \
       --edges-file data/edges.tsv \
       --samples-file data/nodes/samples.txt \
       --feature-files data/nodes/microbes.txt data/nodes/metabolites.txt \
       --slurm
