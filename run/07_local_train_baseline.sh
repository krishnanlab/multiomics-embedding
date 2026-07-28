#!/usr/bin/env sh

# Usage: ./run/07_local_train_baseline.sh
# Trains logistic regression models directly on the raw -omics data (no
# embedding), as a baseline for comparison. A few minutes to run.
exec conda run -n multiomics python scripts/train_baseline_models.py
