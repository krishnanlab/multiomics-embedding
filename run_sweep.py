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

def create_yaml_config(sweep_name, metric, fold):
    if fold is None:
        fold = {'values': [1,2,3,4,5]}
        file_name = f'configs/sweep_config_{sweep_name}.yaml'
    else:
        file_name = f'configs/sweep_config_{sweep_name}_fold_{fold}.yaml'
        fold = {'value': fold}
    config = {
        'program': 'src/sweep.py',
        'name': sweep_name,
        'method': 'random',
        'metric': {
            'name': metric,
            'goal': 'maximize'
        },
        'parameters': {
            'fold': fold,
            'p': {
                'distribution': 'int_uniform',
                'min': 1,
                'max': 25
            },
            'q': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 10
            },
            'g': {
                'values': [0, 1, 2]
            },
            'penalty': {
                'values': ['l1', "l2"]
            },
            'c': {
                'distribution': 'int_uniform',
                'min': 10,
                'max': 300
            }
        },
        'command': [
            'python',
            '${program}',
            '--n2v',
            'OTF',
            '--sweep',
            sweep_name,
            '--filter',
            'from_adelle',
            '${args}',
        ]
    }
    with open(file_name, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    return file_name

def start_sweep(config_file, sweep_name):
    bash_command = ["wandb", "sweep", "-p", sweep_name, config_file]
    result = subprocess.run(bash_command, capture_output=True, text=True)
    output = result.stderr.splitlines()
    for line in output:
        if "ID:" in line:
            sweep_id = line.split()[-1]
            print(f"Started sweep with ID: {sweep_id}")
            return sweep_id
    print(output)

def submit_sweep_jobs(sweep_id, sweep_name, job_dir, num_runs):
    jobsh = f"{sweep_id}.sh"
    with open(os.path.join(job_dir, jobsh), 'w') as jobConn:
        jobConn.write("#!/bin/bash -login\n")
        jobConn.write("#SBATCH --time=3:59:00\n")
        jobConn.write("#SBATCH --mem=512GB\n")
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
        jobConn.write(f"wandb agent -p {sweep_name} -e keenan-manpearl --count 1 {sweep_id}\n")
        jobConn.write("conda deactivate\n")

    os.system(f"sbatch {os.path.join(job_dir, jobsh)}")


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--runs",
                        help="number of models to train in hyperparameter sweep",
                        required=True,
                        type=int)
    parser.add_argument("--model",
                        help="model number to run hyperparameter sweep on if not all",
                        required=False,
                        type=int, 
                        default=None)
    parser.add_argument("--name",
                        help="optional sweep name",
                        required=False,
                        type=str,
                        default=None)
    parser.add_argument("--sweep",
                        help="optional sweep ID to add runs to existing sweep, sweep names must match",
                        required=False,
                        type=str,
                        default=None)
    parser.add_argument("--metric",
                        help="metric to optimize in sweep",
                        required=False,
                        type=str,
                        default='f1_val_avg')
    
    
    args = parser.parse_args()
    # fold to perform sweep on
    model = args.model
    # how many models (hyperparameter combos) to train
    num_runs = args.runs
    metric = args.metric
    # optional sweep name
    if args.name is None:
        sweep_name = f"multiomics_model_{model}"
    else:
        sweep_name = args.name

    # where to write job files
    job_dir = 'run_logs/sweep'
    if not os.path.exists(job_dir):
        os.mkdir(job_dir)    

    # start the sweep if its new 
    if args.sweep is None:
        # create yaml file for model
        file_name = create_yaml_config(sweep_name, metric, model)
        sweep_id = start_sweep(file_name, sweep_name)
    else:
        # get ID if resuming 
        sweep_id = args.sweep
    # submit each run as a job
    submit_sweep_jobs(sweep_id, sweep_name, job_dir, num_runs)
