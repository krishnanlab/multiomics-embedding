"""
Author: Keenan Manpearl
Date: 2026-07-24

Orchestrates a large permutation-testing run (see src/permutation.py):
splits --n-permutations into chunks of --batch-size permutations each
(default 1000/batch - see the slurm-job-sizing skill for how that
number was picked), each run as its own scripts/run_permutation_batch.py
process - local concurrent subprocesses by default, or individual SLURM
jobs (--slurm). Pass --n-batches instead to pick the batch count
directly (overrides --batch-size). Always writes manifest.json, so
scripts/combine_permutations.py has one uniform interface either way.

Run scripts/fit_permutation_test.py first to produce --fitted-state.

--extend PATH grows a previous run into a bigger one without recomputing
it: point --extend at that run's manifest.json, and pass the new (larger)
--n-permutations/--n-batches. numpy's SeedSequence(seed).spawn(n) is
prefix-consistent (spawn(100)[:10] == spawn(10), including the actual
generated streams), so as long as --base-seed matches and every batch is
the same size (keep --n-permutations an exact multiple of a fixed
per-batch size, e.g. always 1000/batch), the first len(existing batches)
batches of the bigger run are byte-identical to the smaller run's - they're
reused as-is, only the new batches beyond that are actually computed.

"""

import os
import json
import math
from argparse import ArgumentParser
from datetime import datetime, timezone

import numpy as np

from job_utils import run_commands_concurrently
import slurm_utils


def build_command(fitted_state: str, n_permutations: int, seed: int, out: str) -> list[str]:
    return [
        "python",
        "scripts/run_permutation_batch.py",
        "--fitted-state",
        fitted_state,
        "--n-permutations",
        str(n_permutations),
        "--seed",
        str(seed),
        "--out",
        out,
    ]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fitted-state", required=True)
    parser.add_argument("--n-permutations", required=False, type=int, default=10000)
    parser.add_argument(
        "--batch-size",
        required=False,
        type=int,
        default=1000,
        help="permutations per batch (default: 1000) - ignored if --n-batches is given",
    )
    parser.add_argument(
        "--n-batches",
        required=False,
        type=int,
        default=None,
        help="explicit batch count - overrides --batch-size if given",
    )
    parser.add_argument(
        "--out", required=True, help="results dir - writes batches/ and manifest.json"
    )
    parser.add_argument(
        "--max_jobs",
        required=False,
        type=int,
        default=4,
        help="local concurrency (ignored if --slurm)",
    )
    parser.add_argument("--base-seed", required=False, type=int, default=0)
    parser.add_argument(
        "--extend",
        required=False,
        default=None,
        help="path to a previous run's manifest.json - reuse its batches "
        "unchanged and only compute the new ones needed to reach the "
        "current --n-permutations/--n-batches (see module docstring)",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="submit each batch as its own SLURM job instead of running "
        "them as local concurrent subprocesses",
    )
    parser.add_argument("--slurm-time", default=slurm_utils.DEFAULT_TIME)
    parser.add_argument("--slurm-mem", default=slurm_utils.DEFAULT_MEM)
    parser.add_argument("--slurm-cpus", type=int, default=slurm_utils.DEFAULT_CPUS)
    args = parser.parse_args()

    n_batches = (
        args.n_batches
        if args.n_batches is not None
        else math.ceil(args.n_permutations / args.batch_size)
    )

    os.makedirs(f"{args.out}/batches", exist_ok=True)

    existing_batches = {}
    if args.extend:
        with open(args.extend) as f:
            prev_manifest = json.load(f)
        if prev_manifest["base_seed"] != args.base_seed:
            parser.error(
                f"--extend manifest used base_seed={prev_manifest['base_seed']}, "
                f"but --base-seed={args.base_seed} - these must match for the "
                "reused batches' seeds to actually line up"
            )
        if prev_manifest["fitted_state"] != args.fitted_state:
            print(
                f"WARNING: --extend manifest was fit from "
                f"{prev_manifest['fitted_state']!r}, differs from --fitted-state "
                f"{args.fitted_state!r}"
            )
        existing_batches = {b["batch_id"]: b for b in prev_manifest["batches"]}

    base, remainder = divmod(args.n_permutations, n_batches)
    batch_sizes = [base + 1 if i < remainder else base for i in range(n_batches)]
    seed_sequences = np.random.SeedSequence(args.base_seed).spawn(n_batches)

    batches = []
    cmds = []
    n_reused = 0
    for i, (size, seed_seq) in enumerate(zip(batch_sizes, seed_sequences)):
        if size == 0:
            continue
        seed = int(seed_seq.generate_state(1)[0])

        if i in existing_batches:
            prev = existing_batches[i]
            if prev["seed"] != seed or prev["n_permutations"] != size:
                raise ValueError(
                    f"batch {i} would be recomputed with a different seed/size "
                    f"than --extend's manifest has (existing: seed={prev['seed']}, "
                    f"n_permutations={prev['n_permutations']}; new: seed={seed}, "
                    f"n_permutations={size}) - the two runs' --n-permutations/"
                    "--n-batches don't use a consistent per-batch size, so "
                    "batches don't align and can't be safely reused"
                )
            batches.append(prev)
            n_reused += 1
            continue

        batch_path = f"{args.out}/batches/batch_{i:04d}.npz"
        job_name = f"perm_batch_{i}"
        command = build_command(args.fitted_state, size, seed, batch_path)

        job_id = None
        slurm_log = None
        if args.slurm:
            job_id = slurm_utils.submit_job(
                command=" ".join(command),
                job_name=job_name,
                log_dir="logging",
                time=args.slurm_time,
                mem=args.slurm_mem,
                cpus=args.slurm_cpus,
            )
            slurm_log = f"logging/slurm-{job_id}_{job_name}.out" if job_id else None
        else:
            cmds.append(command)

        batches.append(
            {
                "batch_id": i,
                "seed": seed,
                "n_permutations": size,
                "batch_path": batch_path,
                "job_id": job_id,
                "slurm_log": slurm_log,
            }
        )

    if n_reused:
        print(f"reused {n_reused} batch(es) unchanged from {args.extend}")

    if not args.slurm:
        run_commands_concurrently(
            commands=cmds,
            max_jobs=args.max_jobs,
            log_file=os.path.join("logs", "run_permutations.log"),
        )

    manifest = {
        "fitted_state": args.fitted_state,
        "observed_tsv": os.path.join(os.path.dirname(args.fitted_state), "observed.tsv"),
        "n_permutations": args.n_permutations,
        "n_batches": n_batches,
        "base_seed": args.base_seed,
        "out_dir": args.out,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "batches": batches,
    }
    manifest_path = f"{args.out}/manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote manifest for {len(batches)} batches to {manifest_path}")
