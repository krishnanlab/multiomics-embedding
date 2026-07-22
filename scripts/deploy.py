"""
Author: Keenan Manpearl
Date: 2026-07-21

This script parses CLI args into an EmbeddingParams and runs one
node2vec+ embedding + classifier deployment pass (see
src/deployment.py's DeploymentRunner) for this study's time/diet
classifiers (see scripts/sweep_setup.py): fits a final model per label on
all available data, with no held-out test set - the "tuning" counterpart
to this is scripts/sweep.py.

"""

from argparse import ArgumentParser

from cli_common import (
    add_embedding_args,
    add_wandb_args,
    require_wandb_project,
    resolve_embedding,
    run_with_optional_wandb,
)
from sweep_setup import build_deployment_runner  # fixes sys.path for src.*
from src.sweep import EmbeddingParams

if __name__ == "__main__":
    parser = ArgumentParser()
    add_embedding_args(parser)
    add_wandb_args(parser)
    parser.add_argument(
        "--out",
        help="directory to save the model/weights/feature-predictions to",
        required=False,
        default=None,
    )

    args = parser.parse_args()
    require_wandb_project(args)

    embedding = resolve_embedding(args)

    params = EmbeddingParams(
        p=args.p,
        q=args.q,
        gamma=args.g,
        dim=args.dim,
        walk_length=args.walk_length,
        window_size=args.window_size,
        n2v_mode=args.n2v,
        seed=args.seed,
        workers=args.workers,
    )

    runner = build_deployment_runner(edg_file=args.edges_file)
    results = run_with_optional_wandb(
        lambda wandb_on: runner.run(
            params, save_to=args.out, embedding=embedding, log_wandb=wandb_on
        ),
        args.sweep,
        not args.no_wandb,
    )
    if args.no_wandb:
        print(results)
