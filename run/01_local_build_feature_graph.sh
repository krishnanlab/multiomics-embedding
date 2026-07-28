#!/usr/bin/env sh

# Usage: ./run/01_local_build_feature_graph.sh
# Builds data/edges.tsv and data/nodes/{samples,microbes,metabolites}.txt
# from raw_data/. Seconds to run - no SLURM variant.
exec conda run -n multiomics python scripts/build_feature_graph.py
