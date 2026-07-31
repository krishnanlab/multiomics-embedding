"""
Author: Keenan Manpearl
Date: 2026-07-22

Generic "run this shell command as a SLURM job" helper - the SLURM
counterpart to job_utils.run_commands_concurrently. Knows nothing about
what command it's running - just fills jobs/template.sh and submits via
sbatch. All cluster-specific detail (partition/qos/account, modules) lives
in jobs/template.sh; edit that for your own HPC setup.

"""

import re
import subprocess
from pathlib import Path

TEMPLATE_PATH = "jobs/template.sh"
DEFAULT_TIME = "00:30:00"
DEFAULT_MEM = "5GB"
DEFAULT_CPUS = 4


def fill_template(
    template: str,
    job_name: str,
    output_log: str,
    command: str,
    time: str,
    mem: str,
    cpus: int,
) -> str:
    return (
        template.replace("{{JOB_NAME}}", job_name)
        .replace("{{OUTPUT_LOG}}", output_log)
        .replace("{{COMMAND}}", command)
        .replace("{{TIME}}", time)
        .replace("{{MEM}}", mem)
        .replace("{{CPUS}}", str(cpus))
    )


def submit_job(
    command: str,
    job_name: str,
    log_dir: str,
    template_path: str = TEMPLATE_PATH,
    time: str = DEFAULT_TIME,
    mem: str = DEFAULT_MEM,
    cpus: int = DEFAULT_CPUS,
    dry_run: bool = False,
) -> "str | None":
    """
    write and (unless dry_run) submit one command as a SLURM job via
    sbatch; returns the SLURM job ID sbatch reports, or None if not
    actually submitted (dry_run, or sbatch isn't available)
    """
    template = Path(template_path).read_text()
    output_log = f"{log_dir}/slurm-%j_{job_name}.out"
    script = fill_template(template, job_name, output_log, command, time, mem, cpus)

    job_file = Path(log_dir) / f"{job_name}.sh"
    job_file.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"--- {job_file} ---\n{script}")
        print(f"(dry run, would run: sbatch {job_file})\n")
        return None

    job_file.write_text(script)
    result = subprocess.run(
        ["sbatch", str(job_file)], check=True, capture_output=True, text=True
    )
    print(result.stdout.strip())
    match = re.search(r"\d+", result.stdout)
    return match.group() if match else None
