#!/usr/bin/env sh

# Usage: ./run/run_all.sh
exec conda run -n multiomics python scripts/submit_all_embeddings.py --data_dir data/emb/from_adelle