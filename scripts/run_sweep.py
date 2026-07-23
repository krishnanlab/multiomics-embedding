"""
Author: Keenan Manpearl
Date: 2024-09-09

This script submits one run of a wandb sweep per parameter combo specified in num_runs

"""

import os
import yaml
import subprocess
import argparse
from job_utils import run_commands_concurrently
import slurm_utils


def get_config_file(name: str) -> str:
    """
    Get a unique file name for starting a new sweep.
    Ensures that different sweeps with the same name have different config files.
    """
    os.makedirs("configs", exist_ok=True)
    fp = f"configs/sweep_config_{name}.yaml"
    count = 1
    while os.path.exists(fp):
        fp = f"configs/sweep_config_{name}_{count}.yaml"
        count += 1
    return fp


def get_distribution(min_val: float) -> str:
    """
    if minimum value is a faction, return uniform distrubtion
    else return int unifrom
    """
    if min_val < 1:
        return "uniform"
    else:
        return "int_uniform"


def create_yaml_config(
    sweep_name: str,
    metric: str,
    p_min: float,
    p_max: float,
    q_min: float,
    q_max: float,
    g_min: int,
    g_max: int,
    edges_file: str,
    samples_file: str,
    feature_files: list[str],
    workers: int,
    inner_cv_folds: int,
    n_iter_search: int,
    num_walks: int,
    walk_length: int,
) -> str:
    """
    write a yaml file with the sweep configuration. edges_file/
    samples_file/feature_files are passed through as-is to scripts/sweep.py
    (required there since this session's node-list validation was added) -
    this script doesn't hardcode which study/graph is being swept, that's
    the caller's job (see run/run_initial_sweep.sh, run/run_joint_sweep.sh).
    workers/inner_cv_folds/n_iter_search/num_walks/walk_length are fixed
    across every trial (not swept) - pass them explicitly rather than
    relying on scripts/sweep.py's own defaults, e.g. to match workers to
    the actual CPUs available (local --max_jobs sharing a machine, or
    per-SLURM-job --slurm-cpus) or to shrink the search for a quick test.
    """
    file_name = get_config_file(sweep_name)
    p_dist = get_distribution(p_min)
    q_dist = get_distribution(q_min)
    config = {
        "program": "scripts/sweep.py",
        "name": sweep_name,
        "method": "random",
        "metric": {"name": metric, "goal": "maximize"},
        "parameters": {
            "p": {"distribution": p_dist, "min": p_min, "max": p_max},
            "q": {"distribution": q_dist, "min": q_min, "max": q_max},
            "g": {"values": list(range(g_min, g_max + 1))},
        },
        "command": [
            "python",
            "${program}",
            "--n2v",
            "OTF",
            "--sweep",
            sweep_name,
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
            "${args}",
        ],
    }
    with open(file_name, "w") as file:
        yaml.dump(config, file, default_flow_style=False)
    return file_name


def start_sweep(config_file: str, sweep_name: str) -> str | None:
    """
    start a new sweep using the parameters specified in config_file
    and return the sweep ID
    """
    # start sweep
    bash_command = ["wandb", "sweep", "-p", sweep_name, config_file]
    result = subprocess.run(bash_command, capture_output=True, text=True)
    # store terminal output
    output = result.stderr.splitlines()
    # get sweep ID
    for line in output:
        if "ID:" in line:
            sweep_id = line.split()[-1]
            print(f"Started sweep with ID: {sweep_id}")
            return sweep_id


def submit_sweep_jobs(
    sweep_id: str,
    sweep_name: str,
    user_name: str,
    num_runs: int,
    max_jobs: int,
    slurm: bool = False,
    slurm_time: str = slurm_utils.DEFAULT_TIME,
    slurm_mem: str = slurm_utils.DEFAULT_MEM,
    slurm_cpus: int = slurm_utils.DEFAULT_CPUS,
) -> None:
    """
    Submit num_runs wandb agent jobs for the given sweep. By default (slurm=
    False - the non-SLURM case) these run as local subprocesses, at most
    max_jobs concurrently, exactly as before. If slurm=True, each run is
    instead submitted as its own SLURM job via scripts/slurm_utils.py (and
    jobs/template.sh - edit that per-cluster, nothing here needs to change);
    max_jobs is unused in that case since SLURM's own queue manages
    concurrency.
    """
    agent_cmd = ["wandb", "agent", "-p", sweep_name, "-e", user_name, "--count", "1", sweep_id]

    if not slurm:
        cmds = [agent_cmd for _ in range(num_runs)]
        run_commands_concurrently(
            commands=cmds,
            max_jobs=max_jobs,
            log_file=os.path.join("logs", f"{sweep_name}.log"),
        )
        return

    for i in range(num_runs):
        slurm_utils.submit_job(
            command=" ".join(agent_cmd),
            job_name=f"{sweep_name}_{i}",
            log_dir="logging",
            time=slurm_time,
            mem=slurm_mem,
            cpus=slurm_cpus,
        )


def number_type(x: str) -> int | float:
    """
    parse a command line argument as an int if possible, else a float
    """
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{x!r} is not a valid int or float")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--username", help="wandb username.", type=str)
    parser.add_argument(
        "--runs",
        help="number of models to train in hyperparameter sweep",
        required=False,
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max_jobs",
        help="number of models to train at one time",
        required=False,
        type=int,
        default=4,
    )
    parser.add_argument(
        "--name",
        help="optional sweep name",
        required=False,
        type=str,
        default="multiomics_embedding",
    )
    parser.add_argument(
        "--sweep",
        help="optional sweep ID to add runs to existing sweep, sweep names must match",
        required=False,
        type=str,
        default=None,
    )
    parser.add_argument(
        "--metric",
        help="metric to optimize in sweep",
        required=False,
        type=str,
        default="emb_score",
    )
    parser.add_argument(
        "--p_min",
        help="minimum p value to test",
        required=False,
        type=number_type,
        default=1,
    )
    parser.add_argument(
        "--p_max",
        help="maximum p value to test",
        required=False,
        type=number_type,
        default=25,
    )
    parser.add_argument(
        "--q_min",
        help="minimum q value to test",
        required=False,
        type=number_type,
        default=0.1,
    )
    parser.add_argument(
        "--q_max",
        help="maximum q value to test",
        required=False,
        type=number_type,
        default=10,
    )
    parser.add_argument(
        "--g_min", help="minimum g value to test", required=False, type=int, default=0
    )
    parser.add_argument(
        "--g_max", help="maximum g value to test", required=False, type=int, default=2
    )
    parser.add_argument(
        "--edges-file", help="edge list to sweep over", required=True
    )
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
        help="submit each wandb agent run as its own SLURM job (via "
        "jobs/template.sh - see scripts/slurm_utils.py) instead of running "
        "them as local concurrent subprocesses",
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

    username = args.username
    sweep_name = args.name
    num_runs = args.runs
    max_jobs = args.max_jobs
    metric = args.metric
    workers = args.workers if args.workers is not None else (
        args.slurm_cpus if args.slurm else 4
    )

    # either start a new sweep if no sweep ID is provided
    if args.sweep is None:
        # create yaml file for model
        p_min = args.p_min
        p_max = args.p_max
        q_min = args.q_min
        q_max = args.q_max
        g_min = args.g_min
        g_max = args.g_max
        file_name = create_yaml_config(
            sweep_name, metric, p_min, p_max, q_min, q_max, g_min, g_max,
            args.edges_file, args.samples_file, args.feature_files, workers,
            args.inner_cv_folds, args.n_iter_search, args.num_walks, args.walk_length,
        )
        sweep_id = start_sweep(file_name, sweep_name)
        if sweep_id is None:
            raise ValueError("Sweep ID not found. Are you logged into wandb?")
    else:
        # or get ID if resuming sweep
        sweep_id = args.sweep
    submit_sweep_jobs(
        sweep_id=sweep_id,
        sweep_name=sweep_name,
        user_name=username,
        num_runs=num_runs,
        max_jobs=max_jobs,
        slurm=args.slurm,
        slurm_time=args.slurm_time,
        slurm_mem=args.slurm_mem,
        slurm_cpus=args.slurm_cpus,
    )
