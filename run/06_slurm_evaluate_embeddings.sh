#!/usr/bin/env sh

# Usage: ./run/06_slurm_evaluate_embeddings.sh
# Same as run/06_local_evaluate_embeddings.sh, but submits the whole
# re-evaluation as one SLURM job instead of running it inline.
# submit_all_embeddings.py has no internal --slurm support (unlike
# run_sweep.py), so this is one job running --max_jobs 4 concurrent
# subprocesses, not one job per embedding - ~2-4 min/embedding (see
# README), so scale --time to however many embeddings ended up cached.
exec conda run -n multiomics python scripts/submit_job.py \
    --job-name evaluate_embeddings \
    --time 08:00:00 \
    -- python scripts/submit_all_embeddings.py \
       --data-dir emb_cache \
       --edges-file data/edges.tsv \
       --samples-file data/nodes/samples.txt \
       --feature-files data/nodes/microbes.txt data/nodes/metabolites.txt \
       --out results/compare_embeddings
