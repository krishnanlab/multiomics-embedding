#!/usr/bin/env sh

# Usage: ./run/local_05_joint_sweep_wandb.sh <wandb_username>
# Random-searches node2vec+ params (p, q, gamma), ranked by the combined
# time+diet validation score (emb_score). Requires `wandb login`. Runs
# trials as local concurrent subprocesses (--max_jobs 4) - see
# run/slurm_05_joint_sweep_wandb.sh to submit each trial as its own SLURM
# job instead.
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <wandb_username>"
  exit 1
fi

USERNAME="$1"

exec conda run -n multiomics \
     python scripts/run_sweep.py \
       --username "$USERNAME" \
       --runs 200 \
       --max_jobs 4 \
       --name joint_optimization_test \
       --edges-file data/edges.tsv \
       --samples-file data/nodes/samples.txt \
       --feature-files data/nodes/microbes.txt data/nodes/metabolites.txt
