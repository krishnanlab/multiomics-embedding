"""
Author: Keenan Manpearl
Date: 2024-09-09

This script parses CLI args into an EmbeddingParams and runs one
node2vec+ embedding + classifier sweep pass (see src/sweep.py's
SweepRunner) for this study's time/diet classifiers (see
scripts/sweep_setup.py), optionally as part of a wandb hyperparameter
sweep (or standalone, see --no-wandb).

"""

from argparse import ArgumentParser

from cli_common import (
    add_embedding_args,
    add_wandb_args,
    require_wandb_project,
    resolve_embedding,
    run_with_optional_wandb,
)
from sweep_setup import build_sweep_runner  # fixes sys.path for src.*
from src.sweep import EmbeddingParams


def main(
    params: EmbeddingParams,
    edg_file: str,
    sweep_name: "str | None",
    save_to: "str | None",
    log_wandb: bool,
    embedding=None,
    inner_cv_folds: int = 10,
    n_iter_search: int = 500,
) -> dict:
    """
    run one node2vec+ embedding and classifier training pass, optionally as
    part of a wandb hyperparameter sweep. Always returns the SweepRunner's
    results dict.
    """
    runner = build_sweep_runner(
        edg_file=edg_file, inner_cv_folds=inner_cv_folds, n_iter_search=n_iter_search
    )
    return run_with_optional_wandb(
        lambda wandb_on: runner.run(
            params, log_wandb=wandb_on, save_models_to=save_to, embedding=embedding
        ),
        sweep_name,
        log_wandb,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    add_embedding_args(parser)
    add_wandb_args(parser)
    parser.add_argument(
        "--save-to",
        help="directory to save the model and results JSON to (default: don't save)",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--inner-cv-folds",
        help="number of inner CV folds for the hyperparameter search within "
        "each outer fold",
        required=False,
        type=int,
        default=10,
    )
    parser.add_argument(
        "--n-iter-search",
        help="number of RandomizedSearchCV candidates to try",
        required=False,
        type=int,
        default=500,
    )

    args = parser.parse_args()
    require_wandb_project(args)

    embedding = resolve_embedding(args)

    params = EmbeddingParams(
        p=args.p,
        q=args.q,
        gamma=args.g,
        dim=args.dim,
        num_walks=args.num_walks,
        walk_length=args.walk_length,
        window_size=args.window_size,
        n2v_mode=args.n2v,
        seed=args.seed,
        workers=args.workers,
    )
    results = main(
        params=params,
        edg_file=args.edges_file,
        sweep_name=args.sweep,
        save_to=args.save_to,
        log_wandb=not args.no_wandb,
        embedding=embedding,
        inner_cv_folds=args.inner_cv_folds,
        n_iter_search=args.n_iter_search,
    )
    if args.no_wandb:
        print(results)
