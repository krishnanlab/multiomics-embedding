import os
import argparse
from pathlib import Path
from job_utils import run_commands_concurrently

def get_emb_params(emb_file):
    '''
    parses filepath and returns p, q, and g
    '''
    params = emb_file.split('_')
    return params[2], params[4], params[6].split('.')[0]

def submit_param_jobs(
    params_list: list[tuple[int,int,int]],
    max_jobs: int,
):
    cmds = []
    for p, q, g in params_list:
        cmds.append([
            "python", "main.py",
            "--name", "all_embeddings",
            "--p", str(p),
            "--q", str(q),
            "--g", str(g)
        ])
    run_commands_concurrently(commands= cmds, 
                              max_jobs= max_jobs, 
                              log_file=os.path.join("logs", f"p{p}_q{q}_g{g}.log")
                            )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",
                   help="Directory containing all embedding files.")
    p.add_argument("--max_jobs", 
                   help="Number of models to train at one time.",
                   type=int, 
                   default=4)
    args = p.parse_args()

    files = os.listdir(args.data_dir)
    params = [get_emb_params(f) for f in files]
    submit_param_jobs(params_list=params, max_jobs=args.max_jobs)