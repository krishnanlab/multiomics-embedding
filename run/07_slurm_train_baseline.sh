#!/usr/bin/env sh

# Usage: ./run/07_slurm_train_baseline.sh
# Same as run/07_local_train_baseline.sh, but submits it as one SLURM job
# instead of running it inline.
exec conda run -n multiomics python scripts/submit_job.py \
    --job-name train_baseline \
    -- python scripts/train_baseline_models.py
