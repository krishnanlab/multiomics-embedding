#!/usr/bin/env sh

# Usage: ./run/08_slurm_train_deployment.sh
# Same as run/08_local_train_deployment.sh, but submits it as one SLURM job
# instead of running it inline.
exec conda run -n multiomics python scripts/submit_job.py \
    --job-name train_deployment \
    -- python scripts/train_deployment_models.py
