#!/usr/bin/env sh

# Usage: ./run/local_07_train_baseline.sh
# Trains logistic regression models directly on the raw -omics data (no
# embedding), as a baseline for comparison. A few minutes to run.
exec conda run -n multiomics python scripts/train_baseline_models.py
