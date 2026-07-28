#!/usr/bin/env sh

# Usage: ./run/09_local_train_deployment_for_permutations.sh
# Same underlying script as run/08_local_train_deployment.sh, but with a
# fixed (non-dated) --out so the permutation-testing scripts (09-12) have a
# stable path to point at: results/best/*_logging.txt combines both labels
# in one file (pre-refactor format), which scripts/fit_permutation_test.py
# can't parse correctly - this writes the current, correctly-separated
# per-label format instead. ~3 minutes to run.
exec conda run -n multiomics python scripts/train_deployment_models.py \
    --out results/deployment_for_permutations
