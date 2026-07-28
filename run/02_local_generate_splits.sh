#!/usr/bin/env sh

# Usage: ./run/02_local_generate_splits.sh
# Builds data/node_splits.tsv (outer CV fold assignment) from
# raw_data/sample_breakdown.csv. Seconds to run - no SLURM variant.
exec conda run -n multiomics python scripts/generate_splits.py
