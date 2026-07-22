"""
Author: Keenan Manpearl
Date: 2026-07-21

CLI argument wiring shared by scripts/sweep.py and scripts/deploy.py:
both need the same node2vec+ embedding-parameter flags, the same
required edge-list/samples/feature-file inputs, the same optional
pre-built-embedding override, and the same validate-then-resolve
sequence before doing anything else. Study-agnostic - just argparse
plumbing and a thin wrapper around src/validation.py.

"""

import sys
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Callable

import pandas as pd
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.validation import validate_node_lists


def add_embedding_args(parser: ArgumentParser) -> None:
    """add the CLI args shared by scripts/sweep.py and scripts/deploy.py"""
    parser.add_argument("--edges-file", help="edge list to embed", required=True)
    parser.add_argument(
        "--samples-file",
        help="path to a newline-separated list of sample node IDs",
        required=True,
    )
    parser.add_argument(
        "--feature-files",
        help="one or more newline-separated lists of feature node IDs",
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--embedding-file",
        help="use this pre-built embedding directly instead of generating one - "
        "if set, -p/-q/-g/--dim/--walk-length/--window-size are ignored",
        required=False,
        default=None,
    )
    parser.add_argument("--p", help="node2vec p parameter", required=True, type=float)
    parser.add_argument("--q", help="node2vec q parameter", required=True, type=float)
    parser.add_argument(
        "--g", help="node2vec gamma parameter", required=True, type=float
    )
    parser.add_argument(
        "--dim", help="embedding dimensionality", required=False, type=int, default=128
    )
    parser.add_argument(
        "--walk-length",
        help="node2vec random walk length",
        required=False,
        type=int,
        default=80,
    )
    parser.add_argument(
        "--window-size",
        help="skip-gram context window size",
        required=False,
        type=int,
        default=10,
    )
    parser.add_argument(
        "--workers",
        help="thread count for embedding generation - match this to however many "
        "CPUs are actually available (e.g. a SLURM job's --cpus-per-task)",
        required=False,
        type=int,
        default=4,
    )
    parser.add_argument(
        "--seed", help="seed for reproducibility", required=False, type=int, default=42
    )
    parser.add_argument(
        "--n2v",
        help="node2vec graph type: effects time and memory usage",
        required=False,
        choices=["OTF", "Pre"],
        default="OTF",
    )


def add_wandb_args(parser: ArgumentParser) -> None:
    """add the --sweep/--no-wandb args shared by scripts/sweep.py and scripts/deploy.py"""
    parser.add_argument(
        "--sweep",
        help="wandb project name (required unless --no-wandb)",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--no-wandb",
        help="skip wandb logging and just print/return results",
        action="store_true",
    )


def require_wandb_project(args: Namespace) -> None:
    """call after parsing args that used add_wandb_args - errors like argparse would"""
    if not args.no_wandb and args.sweep is None:
        raise SystemExit("error: --sweep is required unless --no-wandb is set")


def run_with_optional_wandb(
    run_fn: Callable[[bool], dict], sweep_name: "str | None", log_wandb: bool
) -> dict:
    """
    call run_fn(log_wandb=...), optionally wrapped in a wandb run - shared
    by scripts/sweep.py and scripts/deploy.py so the wandb.init/finish
    wrapping isn't duplicated between them.
    """
    if not log_wandb:
        return run_fn(False)
    with wandb.init(project=sweep_name):
        results = run_fn(True)
    wandb.finish()
    return results


def resolve_embedding(args: Namespace) -> "pd.DataFrame | None":
    """
    Validate the samples/feature-files against the graph (and against
    --embedding-file, if given) - raises if a listed node is missing from
    either, warns if the graph/embedding has nodes not listed anywhere.
    Returns the loaded embedding if --embedding-file was given (and warns
    that the embedding-generation flags are being ignored), else None -
    meaning the caller should generate/load-from-cache one as usual.
    """
    list_files = {"samples": args.samples_file}
    list_files.update(
        {f"features[{i}]": path for i, path in enumerate(args.feature_files)}
    )

    embedding = None
    embedding_nodes = None
    if args.embedding_file:
        embedding = pd.read_csv(args.embedding_file, sep="\t", index_col=0)
        embedding_nodes = set(embedding.index.astype(str))

    validate_node_lists(args.edges_file, list_files, embedding_nodes)

    if embedding is not None:
        warnings.warn(
            "--embedding-file was given: -p/-q/-g/--dim/--walk-length/"
            "--window-size/--workers are being ignored."
        )
    return embedding
