'''
Author: Keenan Manpearl
Date: 2024-09-09

This script submits one run of a wandb sweep
per array specified in num_runs
It is used within start_sweep.sb

'''

import os
import yaml
import subprocess

def create_yaml_config(fold):
    config = {
        'program': '/mnt/home/f0106093/Projects/multiomics-embedding/src/sweep.py',
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
                'values': [0.0, 0.01, 0.1, 0.5, 0.7]
            },
            'penalty': {
                'values': ["l1", "l2"]
            },
            'c': {
                'values': [1, 10, 100, 1000]
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
    output = result.stdout.splitlines()
    for line in output:
        if "ID:" in line:
            sweep_id = line.split()[-1]
            return sweep_id

def submit_sweep_jobs(sweep_id, fold, job_dir,num_runs):
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
        jobConn.write(f"wandb agent -p multiomics-fold-{fold} -e keenan-manpearl --count 1 {sweep_id}\n")
        jobConn.write("conda deactivate\n")

    os.system(f"sbatch {os.path.join(job_dir, jobsh)}")


if __name__ == '__main__':

    num_runs = 2
    job_dir = '/mnt/home/f0106093/Projects/multiomics-embedding/run_logs/sweep'
    if not os.path.exists(job_dir):
        os.mkdir(job_dir)    

    for fold in [1, 2, 3, 4, 5]:
        file_name = create_yaml_config(fold)
        sweep_id = start_sweep(file_name, fold)
        submit_sweep_jobs(sweep_id, fold, job_dir,num_runs)
