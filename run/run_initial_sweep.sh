#!/usr/bin/env sh

# Usage: ./run_initial_sweep.sh <wandb_username>
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <wandb_username>"
  exit 1
fi

USERNAME="$1"

exec conda run -n multiomics \
     python run/run_sweep.py \
       --username "$USERNAME" \
       --runs 100 \
       --max_jobs 4 \
       --name time_point \
       --metric time_avg_val_f1 \
       --p_max 100 \
       --q_max 100