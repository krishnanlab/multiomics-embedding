"""
Author: Keenan Manpearl
Date: 2026-07-24

Thin CLI around slurm_utils.submit_job: submit one arbitrary shell command
as a single SLURM job. For pipeline steps with no internal --slurm/per-trial
parallelism of their own (scripts/submit_all_embeddings.py,
train_baseline_models.py, train_deployment_models.py) - the whole script
just becomes one batch job instead of running inline.

    python scripts/submit_job.py --job-name my_job -- python scripts/foo.py --bar baz

"""

import argparse

import slurm_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--log-dir", default="logging")
    parser.add_argument("--template", default=slurm_utils.TEMPLATE_PATH)
    parser.add_argument("--time", default=slurm_utils.DEFAULT_TIME)
    parser.add_argument("--mem", default=slurm_utils.DEFAULT_MEM)
    parser.add_argument("--cpus", type=int, default=slurm_utils.DEFAULT_CPUS)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the job script instead of submitting"
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="command to run, after --"
    )
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("no command given (pass it after --)")

    slurm_utils.submit_job(
        command=" ".join(command),
        job_name=args.job_name,
        log_dir=args.log_dir,
        template_path=args.template,
        time=args.time,
        mem=args.mem,
        cpus=args.cpus,
        dry_run=args.dry_run,
    )
