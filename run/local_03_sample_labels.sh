#!/usr/bin/env sh

# Usage: ./run/local_03_sample_labels.sh
# Builds data/time_labels.tsv and data/diet_labels.tsv from raw_data/.
# Seconds to run - no SLURM variant.
exec conda run -n multiomics python scripts/sample_labels.py
