import subprocess
import time
import logging
from pathlib import Path

def run_commands_concurrently(
    commands: list[list[str]],
    max_jobs: int,
    log_file: Path,
    poll_interval: float = 30.0
):

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    active = {}  # {pid : (proc, idx)}

    def _cleanup_finished():
        for pid, (proc, idx) in list(active.items()):
            if proc.poll() is not None:
                out, err = proc.communicate()
                logging.info(f"[#{idx}] PID {pid} exited {proc.returncode}")
                if out:
                    logging.info(f"[#{idx}] stdout:\n{out.strip()}")
                if err:
                    logging.error(f"[#{idx}] stderr:\n{err.strip()}")
                active.pop(pid)

    for idx, cmd in enumerate(commands):
        while len(active) >= max_jobs:
            time.sleep(poll_interval)
            _cleanup_finished()

        print("Launching:", cmd)
        assert all(arg is not None for arg in cmd), f"Found None in {cmd!r}"
        
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE,
                   text=True
        )
        active[proc.pid] = (proc, idx)
        logging.info(f"[#{idx}] START PID {proc.pid}: {' '.join(cmd)}")

    while active:
        time.sleep(poll_interval)
        _cleanup_finished()

    logging.info("All commands finished.")