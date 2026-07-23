"""
Author: Keenan Manpearl
Date: 2026-07-23

Non-wandb counterpart to scripts/run_sweep.py: random-searches node2vec+
parameters (p, q, gamma) the same way (p/q uniform if their range's minimum
is a fraction, else int-uniform; g uniform over the integer range - see
run_sweep.py's get_distribution), but for anyone without a wandb account -
each trial runs scripts/sweep.py directly with --no-wandb --save-to <out>
instead of going through a wandb sweep/agent, so results land as local
results JSONs (one per trial, see BaseRunner._save_results_json) rather
than being logged to wandb. Same local-subprocess/--slurm submission choice
as scripts/run_sweep.py (job_utils.py / scripts/slurm_utils.py).

Run this script directly:

    python scripts/run_sweep_local.py --runs 20 --out results/local_sweep \\
        --edges-file data/edges.tsv --samples-file data/nodes/samples.txt \\
        --feature-files data/nodes/microbes.txt data/nodes/metabolites.txt

"""

import os
import random
import argparse

from job_utils import run_commands_concurrently
from run_sweep import get_distribution, number_type
import slurm_utils


def sample_value(min_val: float, max_val: float) -> "int | float":
    """sample one p/q value, matching run_sweep.py's get_distribution semantics"""
    if get_distribution(min_val) == "uniform":
        return random.uniform(min_val, max_val)
    return random.randint(int(min_val), int(max_val))


def build_command(
    edges_file: str,
    samples_file: str,
    feature_files: list[str],
    workers: int,
    inner_cv_folds: int,
    n_iter_search: int,
    num_walks: int,
    walk_length: int,
    out_dir: str,
    p: float,
    q: float,
    g: int,
) -> list[str]:
    """the scripts/sweep.py invocation for one trial - --no-wandb, no wandb project needed"""
    return [
        "python",
        "scripts/sweep.py",
        "--n2v",
        "OTF",
        "--edges-file",
        edges_file,
        "--samples-file",
        samples_file,
        "--feature-files",
        *feature_files,
        "--workers",
        str(workers),
        "--inner-cv-folds",
        str(inner_cv_folds),
        "--n-iter-search",
        str(n_iter_search),
        "--num-walks",
        str(num_walks),
        "--walk-length",
        str(walk_length),
        "--save-to",
        out_dir,
        "--no-wandb",
        "--p",
        str(p),
        "--q",
        str(q),
        "--g",
        str(g),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        help="number of embedding spaces to try",
        required=False,
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max_jobs",
        help="number of trials to run at one time (ignored if --slurm)",
        required=False,
        type=int,
        default=4,
    )
    parser.add_argument(
        "--out",
        help="directory each trial saves its model/results JSON to",
        required=True,
    )
    parser.add_argument(
        "--p_min", help="minimum p value to test", required=False, type=number_type, default=1
    )
    parser.add_argument(
        "--p_max", help="maximum p value to test", required=False, type=number_type, default=25
    )
    parser.add_argument(
        "--q_min", help="minimum q value to test", required=False, type=number_type, default=0.1
    )
    parser.add_argument(
        "--q_max", help="maximum q value to test", required=False, type=number_type, default=10
    )
    parser.add_argument(
        "--g_min", help="minimum g value to test", required=False, type=int, default=0
    )
    parser.add_argument(
        "--g_max", help="maximum g value to test", required=False, type=int, default=2
    )
    parser.add_argument("--edges-file", help="edge list to sweep over", required=True)
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
        "--slurm",
        help="submit each trial as its own SLURM job (via jobs/template.sh "
        "- see scripts/slurm_utils.py) instead of running them as local "
        "concurrent subprocesses",
        action="store_true",
    )
    parser.add_argument(
        "--slurm-time", default=slurm_utils.DEFAULT_TIME, help="SLURM --time per run"
    )
    parser.add_argument(
        "--slurm-mem", default=slurm_utils.DEFAULT_MEM, help="SLURM --mem per run"
    )
    parser.add_argument(
        "--slurm-cpus",
        type=int,
        default=slurm_utils.DEFAULT_CPUS,
        help="SLURM --cpus-per-task per run",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="thread count each trial's scripts/sweep.py uses (embedding "
        "generation + CV search) - defaults to --slurm-cpus when --slurm is "
        "set (matching the job's own allocation), else 4",
    )
    parser.add_argument(
        "--inner-cv-folds",
        type=int,
        default=10,
        help="inner CV folds each trial's scripts/sweep.py uses (see "
        "SweepRunner's own default)",
    )
    parser.add_argument(
        "--n-iter-search",
        type=int,
        default=500,
        help="RandomizedSearchCV candidates each trial's scripts/sweep.py tries",
    )
    parser.add_argument(
        "--num-walks",
        type=int,
        default=10,
        help="node2vec random walks per node for each trial's embedding",
    )
    parser.add_argument(
        "--walk-length",
        type=int,
        default=80,
        help="node2vec random walk length for each trial's embedding",
    )

    args = parser.parse_args()
    workers = args.workers if args.workers is not None else (
        args.slurm_cpus if args.slurm else 4
    )

    trials = [
        (
            sample_value(args.p_min, args.p_max),
            sample_value(args.q_min, args.q_max),
            random.randint(args.g_min, args.g_max),
        )
        for _ in range(args.runs)
    ]

    if not args.slurm:
        cmds = [
            build_command(
                args.edges_file, args.samples_file, args.feature_files, workers,
                args.inner_cv_folds, args.n_iter_search, args.num_walks, args.walk_length,
                args.out, p, q, g,
            )
            for p, q, g in trials
        ]
        run_commands_concurrently(
            commands=cmds,
            max_jobs=args.max_jobs,
            log_file=os.path.join("logs", "run_sweep_local.log"),
        )
    else:
        for i, (p, q, g) in enumerate(trials):
            command = build_command(
                args.edges_file, args.samples_file, args.feature_files, workers,
                args.inner_cv_folds, args.n_iter_search, args.num_walks, args.walk_length,
                args.out, p, q, g,
            )
            slurm_utils.submit_job(
                command=" ".join(command),
                job_name=f"local_sweep_{i}",
                log_dir="logging",
                time=args.slurm_time,
                mem=args.slurm_mem,
                cpus=args.slurm_cpus,
            )
