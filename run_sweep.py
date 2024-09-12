'''
Author: Keenan Manpearl
Date: 2024-09-09

This script submits one run of a wandb sweep
per array specified in num_runs
and performs the sweep on the specified fold

'''

import os
import yaml
import subprocess
from argparse import ArgumentParser

def create_yaml_config(fold):
    config = {
        'program': 'src/sweep.py',
        'name': f'multiomics-fold-{fold}',
        'method': 'grid',
        'metric': {
            'name': 'val_balanced_accuracy_avg',
            'goal': 'maximize'
        },
        'parameters': {
            'p': {
                'values': [0.01, 0.1, 1, 10, 100]
            },
            'q': {
                'values': [0.01, 0.1, 1, 10, 100]
            },
            'g': {
                'values': [1]
            },
            'penalty': {
                'values': ["l1", "l2"]
            },
            'c': {
                'values': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
            },
            'filter': {
                'values': ['Tau', 'Missingness']
            }
        },
        'command': [
            'python',
            '${program}',
            '--fold',
            fold,
            '--n2v',
            'Pre',
            '${args}'
        ]
    }
    file_name = f'configs/sweep_config_{fold}.yaml'
    with open(file_name, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    return(file_name)

def start_sweep(config_file, fold):
    bash_command = ["wandb", "sweep", "-p", f"multiomics-fold-{fold}", config_file]
    result = subprocess.run(bash_command, capture_output=True, text=True)
    output = result.stderr.splitlines()
    for line in output:
        if "ID:" in line:
            sweep_id = line.split()[-1]
            print(f"Started sweep with ID: {sweep_id}")
            return sweep_id

def submit_sweep_jobs(sweep_id, fold, job_dir, num_runs):
    jobsh = f"{sweep_id}.sh"
    with open(os.path.join(job_dir, jobsh), 'w') as jobConn:
        jobConn.write("#!/bin/bash -login\n")
        jobConn.write("#SBATCH --time=3:59:00\n")
        jobConn.write("#SBATCH --mem=400GB\n")
        jobConn.write("#SBATCH --nodes=1\n")
        jobConn.write("#SBATCH --cpus-per-task=4\n")
        jobConn.write(f"#SBATCH --job-name={sweep_id}\n")
        jobConn.write(f"#SBATCH --output={os.path.join(job_dir, sweep_id)}_%A_%a.out\n")
        jobConn.write(f"#SBATCH -e {os.path.join(job_dir, sweep_id)}_%A_%a.err\n")
        jobConn.write("#SBATCH --account=wang-krishnan\n")
        jobConn.write(f"#SBATCH --array=0-{num_runs-1} \n")
        jobConn.write("module purge\n")
        jobConn.write("module load Conda/3\n")
        jobConn.write("conda activate multiomics\n")
        jobConn.write(f"wandb agent -p multiomics-fold-{fold} -e keenan-manpearl --count 1 {sweep_id}\n")
        jobConn.write("conda deactivate\n")

    os.system(f"sbatch {os.path.join(job_dir, jobsh)}")


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--runs",
                        help="number of models to train in hyperparameter sweep",
                        required=True,
                        type=int)
    parser.add_argument("--fold",
                        help="fold to run hyperparameter sweep on",
                        required=True,
                        type=int)
    
    args = parser.parse_args()
    # fold to perform sweep on
    fold = args.fold
    # how many models (hyperparameter combos) to train
    num_runs = args.runs

    # where to write job files
    job_dir = 'run_logs/sweep'
    if not os.path.exists(job_dir):
        os.mkdir(job_dir)    

    # create yaml file for fold
    file_name = create_yaml_config(fold)
    # start sweep and get ID
    sweep_id = start_sweep(file_name, fold)
    # submit each run as a job
    submit_sweep_jobs(sweep_id, fold, job_dir,num_runs)
