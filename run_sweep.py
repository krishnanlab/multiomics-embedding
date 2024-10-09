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
from typing import Union 


def get_config_file(name):
    '''
    Get a unique file name for starting a new sweep.
    Ensures that different sweeps with the same name have different config files. 
    '''
    fp = f'configs/sweep_config_{name}.yaml'
    count = 1
    while os.path.exists(fp):
        fp = f'configs/sweep_config_{name}_{count}.yaml'
        count += 1
    return fp


def get_distribution(min_val):
    '''
    if minimum value is a faction, return uniform distrubtion
    else return int unifrom 
    '''
    if min_val < 1:
        return 'uniform'
    else:
        return 'int_uniform'


def create_yaml_config(sweep_name, metric, p_min, p_max, q_min, q_max, g_min, g_max):
    '''
    write a yaml file with the sweep configuration
    '''
    file_name = get_config_file(sweep_name)
    p_dist = get_distribution(p_min)
    q_dist = get_distribution(q_min)
    config = {
        'program': 'src/sweep.py',
        'name': sweep_name,
        'method': 'random',
        'metric': {
            'name': metric,
            'goal': 'maximize'
        },
        'parameters': {
            'p': {
                'distribution': p_dist,
                'min': p_min,
                'max': p_max
            },
            'q': {
                'distribution': q_dist,
                'min': q_min,
                'max': q_max
            },
            'g': {
                'values': range(g_min, g_max+1)
            }
        },
        'command': [
            'python',
            '${program}',
            '--n2v',
            'OTF',
            '--sweep',
            sweep_name,
            '${args}',
        ]
    }
    with open(file_name, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    return file_name

def start_sweep(config_file: str, sweep_name: str):
    '''
    start a new sweep using the parameters specified in config_file
    and return the sweep ID
    '''
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

def submit_sweep_jobs(sweep_id: str, sweep_name: str, num_runs: int):
    '''
    write and submit a slurm array job
    '''
    # where to write job files
    job_dir = f'/mnt/scratch/f0106093/multiomics_embedding/run_logs/sweep/{sweep_id}'
    if not os.path.exists(job_dir):
        os.mkdir(job_dir) 

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
                        required=False,
                        type=int,
                        default=100)
    parser.add_argument("--name",
                        help="optional sweep name",
                        required=False,
                        type=str,
                        default='multiomics_embedding')
    parser.add_argument("--sweep",
                        help="optional sweep ID to add runs to existing sweep, sweep names must match",
                        required=False,
                        type=str,
                        default=None)
    parser.add_argument("--metric",
                        help="metric to optimize in sweep",
                        required=False,
                        type=str,
                        default='emb_score')
    parser.add_argument("--p_min",
                        help="minimum p value to test",
                        required=False,
                        type=Union[int, float],
                        default=1)
    parser.add_argument("--p_max",
                        help="maximum p value to test",
                        required=False,
                        type=Union[int, float],
                        default=25)
    parser.add_argument("--q_min",
                        help="minimum q value to test",
                        required=False,
                        type=Union[int, float],
                        default=0.1)
    parser.add_argument("--q_max",
                        help="maximum q value to test",
                        required=False,
                        type=Union[int, float],
                        default=10)
    parser.add_argument("--g_min",
                        help="minimum g value to test",
                        required=False,
                        type=int,
                        default=0)
    parser.add_argument("--g_max",
                        help="maximum g value to test",
                        required=False,
                        type=int,
                        default=2)

    
    
    args = parser.parse_args()

    sweep_name = args.name
    # how many hyperparameter combos to try
    num_runs = args.runs
    # metric to optimize during sweep
    metric = args.metric
   

    # either start a new sweep if no sweep ID is provided 
    if args.sweep is None:
        # create yaml file for model
        p_min = args.p_min
        p_max = args.p_max
        q_min = args.q_min
        q_max = args.q_max
        g_min = args.g_min
        g_max = args.g_max
        file_name = create_yaml_config(sweep_name, metric, p_min, p_max, q_min, q_max, g_min, g_max)
        sweep_id = start_sweep(file_name, sweep_name)
    else:
        # or get ID if resuming sweep
        sweep_id = args.sweep
    # submit each run as a slurm job
    submit_sweep_jobs(sweep_id, sweep_name, num_runs)
