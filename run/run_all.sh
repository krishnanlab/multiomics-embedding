#!/usr/bin/env sh

# Usage: ./run/run_all.sh
exec conda run -n multiomics python scripts/submit_all_embeddings.py \
    --data-dir emb_cache \
    --edges-file data/edges.tsv \
    --samples-file data/nodes/samples.txt \
    --feature-files data/nodes/microbes.txt data/nodes/metabolites.txt \
    --out results/compare_embeddings