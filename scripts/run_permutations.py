"""
Author: Keenan Manpearl
Date: 2026-07-24

Orchestrates a large permutation-testing run (see src/permutation.py):
splits --n-permutations into --n-batches chunks, each run as its own
scripts/run_permutation_batch.py process - local concurrent subprocesses
by default, or individual SLURM jobs (--slurm). Always writes
manifest.json, so scripts/combine_permutations.py has one uniform
interface either way.

Run scripts/fit_permutation_test.py first to produce --fitted-state.

"""

import os
import json
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
    parser.add_argument("--n-batches", required=False, type=int, default=100)
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
        "--slurm",
        action="store_true",
        help="submit each batch as its own SLURM job instead of running "
        "them as local concurrent subprocesses",
    )
    parser.add_argument("--slurm-time", default=slurm_utils.DEFAULT_TIME)
    parser.add_argument("--slurm-mem", default=slurm_utils.DEFAULT_MEM)
    parser.add_argument("--slurm-cpus", type=int, default=slurm_utils.DEFAULT_CPUS)
    args = parser.parse_args()

    os.makedirs(f"{args.out}/batches", exist_ok=True)

    base, remainder = divmod(args.n_permutations, args.n_batches)
    batch_sizes = [base + 1 if i < remainder else base for i in range(args.n_batches)]
    seed_sequences = np.random.SeedSequence(args.base_seed).spawn(args.n_batches)

    batches = []
    cmds = []
    for i, (size, seed_seq) in enumerate(zip(batch_sizes, seed_sequences)):
        if size == 0:
            continue
        seed = int(seed_seq.generate_state(1)[0])
        npy_path = f"{args.out}/batches/batch_{i:04d}.npy"
        job_name = f"perm_batch_{i}"
        command = build_command(args.fitted_state, size, seed, npy_path)

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
                "npy_path": npy_path,
                "job_id": job_id,
                "slurm_log": slurm_log,
            }
        )

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
        "n_batches": args.n_batches,
        "base_seed": args.base_seed,
        "out_dir": args.out,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "batches": batches,
    }
    manifest_path = f"{args.out}/manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote manifest for {len(batches)} batches to {manifest_path}")
