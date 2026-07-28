#!/usr/bin/env sh

# Usage: ./run/06_local_evaluate_embeddings.sh
# Re-evaluates every embedding cached by the two sweeps (emb_cache/) via a
# nested-CV pass, so they can be compared and the top performers selected.
# Runs each embedding as a local concurrent subprocess (--max_jobs 4).
exec conda run -n multiomics python scripts/submit_all_embeddings.py \
    --data-dir emb_cache \
    --edges-file data/edges.tsv \
    --samples-file data/nodes/samples.txt \
    --feature-files data/nodes/microbes.txt data/nodes/metabolites.txt \
    --out results/compare_embeddings
