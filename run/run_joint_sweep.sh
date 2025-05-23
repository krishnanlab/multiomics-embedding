#!/usr/bin/env sh

# Usage: ./run/run_joint_sweep.sh <wandb_username>
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <wandb_username>"
  exit 1
fi

USERNAME="$1"

exec conda run -n multiomics \
     python src/run_sweep.py \
       --username "$USERNAME" \
       --runs 200 \
       --max_jobs 4 \
       --name joint_optimization_test