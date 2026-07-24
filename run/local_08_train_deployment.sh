#!/usr/bin/env sh

# Usage: ./run/local_08_train_deployment.sh
# Fits a final deployment model per classifier for each of the 7 curated
# "best" embedding spaces, and z-scores the resulting feature predictions.
# ~3 minutes to run (embeddings are already cached - see README).
exec conda run -n multiomics python scripts/train_deployment_models.py
