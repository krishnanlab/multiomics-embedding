"""
Author: Keenan Manpearl
Date: 2024-09-09

Trains a classifier on all available samples per fold and predicts on
features (microbial/metabolite), for this study's 7 curated "best"
embeddings. Per-embedding training is src/deployment.py's DeploymentRunner
(same as scripts/deploy.py) - this script's job is just the curated-list
loop and z-scoring aggregation across the 7.

Each embedding's outputs land in <out>/<tag>/ (tag = the same short hex ID
used as the file prefix in results/best, e.g. "wcksnlsg" - see EMBEDDINGS
below), passed through to DeploymentRunner.run()'s tag= kwarg so file names
inside that directory don't need to repeat the embedding's full p/q/g/dim/...
cache tag.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from argparse import ArgumentParser
from datetime import datetime
import gzip
import shutil
import warnings
import os

from sklearn.exceptions import ConvergenceWarning

from src.sweep import EmbeddingParams
from src.zscoring import FeatureZScorer
from sweep_setup import build_deployment_runner, DEFAULT_EDG_FILE

warnings.filterwarnings("ignore", category=ConvergenceWarning)


SEED = 22
MAX_ITER = 100
N_MODELS = 500
SCORING = ["f1", "balanced_accuracy", "accuracy"]

# these 7 embeddings predate the current emb_cache/ naming convention
# (EmbeddingParams.cache_tag) - they were cached under this simpler,
# gzipped name before dim/walk_length/window_size were part of the tag
LEGACY_EMB_DIR = "data/emb"


def _migrate_legacy_cache(
    params: EmbeddingParams, edg_file: str, emb_cache_dir: str = "emb_cache"
) -> None:
    """Copy a curated embedding from its legacy cache path to the current cache_tag path, if not already there - avoids regenerating it."""
    new_path = Path(emb_cache_dir) / f"emb_{params.cache_tag(edg_file)}.tsv"
    if new_path.exists():
        return
    legacy_path = Path(LEGACY_EMB_DIR) / f"emb_p_{params.p}_q_{params.q}_g_{params.gamma}.tsv.gz"
    if not legacy_path.exists():
        return
    new_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(legacy_path, "rb") as src, open(new_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

# curated "best" embedding spaces from the hyperparameter sweep - each
# already cached at data/emb/emb_p_{p}_q_{g}.tsv.gz
EMBEDDINGS = {
    "wcksnlsg": {"p": 19.0, "q": 9.122152261131532, "g": 1},
    "ai9n4jxs": {"p": 0.8055551041134607, "q": 0.1, "g": 1},
    "7o4yga2v": {"p": 0.5, "q": 1.895944090041435, "g": 1},
    "21tdsqsa": {"p": 1.0795506927238254, "q": 8.383911078685804, "g": 1},
    "q2gzu1o3": {"p": 19.0, "q": 8.483911078685804, "g": 2},
    "8lofhbbf": {"p": 7.305688086564288, "q": 7.517332462471247, "g": 2},
    "qb4y98x0": {"p": 5.5, "q": 9.010757520712524, "g": 1},
}


def setup_output_dir(out_dir: "str | None") -> str:
    """Create output directory if it doesn't exist."""
    if out_dir is None:
        current_date = datetime.now().strftime("%Y-%m-%d")
        out_dir = f"results/best_{current_date}"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def main(p: float, q: float, g: int, out_dir: str, tag: str) -> None:
    """
    train deployment models for one embedding space (via
    DeploymentRunner), log CV results, and z-score feature predictions
    """
    print(tag)

    params = EmbeddingParams(p=p, q=q, gamma=g, seed=SEED)
    _migrate_legacy_cache(params, DEFAULT_EDG_FILE)
    runner = build_deployment_runner(
        cv_max_iter=MAX_ITER, n_iter_search=N_MODELS, scoring=SCORING, refit="f1"
    )
    results = runner.run(params, save_to=out_dir, tag=tag)

    zscorer = FeatureZScorer.from_files(
        {
            "microbes": "data/nodes/microbes.txt",
            "metabolites": "data/nodes/metabolites.txt",
        }
    )
    for label_name in runner.labels:
        feature_predictions = results[label_name]["feature_predictions"]
        if feature_predictions is not None:
            zscorer.score_and_save(feature_predictions, f"{out_dir}/{tag}", tag, label_name)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--out",
        help="output directory to save files",
        required=False,
        type=str,
        default=None,
    )
    args = parser.parse_args()
    out_dir = setup_output_dir(args.out)

    for model, embedding_params in EMBEDDINGS.items():
        main(
            p=embedding_params["p"],
            q=embedding_params["q"],
            g=embedding_params["g"],
            out_dir=out_dir,
            tag=model,
        )
