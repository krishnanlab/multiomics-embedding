#!/usr/bin/env sh

# Usage: ./run/run_all.sh
exec conda run -n multiomics python src/submit_all_embeddings.py --data_dir data/emb/from_adelle