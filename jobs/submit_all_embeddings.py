import os 
import time
import subprocess


def get_emb_params(emb_file):
    '''
    parses filepath and returns p, q, and g
    '''
    params = emb_file.split('_')
    return params[2], params[4], params[6].split('.')[0]

def submit_job(p, q, g):
    '''
    write and submit a slurm array job for given parameters
    '''
    # where to write job files
    job_dir = f'/mnt/scratch/f0106093/multiomics_embedding/run_logs/all_emb'
    if not os.path.exists(job_dir):
        os.mkdir(job_dir) 
    tag = f'{p}_{q}_{g}'
    jobsh = f"{tag}.sh"
    with open(os.path.join(job_dir, jobsh), 'w') as jobConn:
        jobConn.write("#!/bin/bash -login\n")
        jobConn.write("#SBATCH --time=0:59:00\n")
        jobConn.write("#SBATCH --mem=128GB\n")
        jobConn.write("#SBATCH --nodes=1\n")
        jobConn.write(f"#SBATCH --job-name=all_emb\n")
        jobConn.write(f"#SBATCH --output={os.path.join(job_dir, tag)}_%A_%a.out\n")
        jobConn.write(f"#SBATCH -e {os.path.join(job_dir, tag)}_%A_%a.err\n")
        jobConn.write("#SBATCH --account=wang-krishnan\n")
        jobConn.write("module purge\n")
        jobConn.write("module load Conda/3\n")
        jobConn.write("conda activate multiomics\n")
        jobConn.write("cd /mnt/home/f0106093/Projects/multiomics-embedding\n")
        jobConn.write(f"python main.py --name all_embeddings --p {p} --q {q} --g {g} \n")
        jobConn.write("conda deactivate\n")

    os.system(f"sbatch {os.path.join(job_dir, jobsh)}")


if __name__ == '__main__':
    dir = 'data/emb/from_adelle'
    files = os.listdir(dir)
    for file in files:
        submit_job(*get_emb_params(file))
        # Check how many jobs on queue at once
        njobs = subprocess.check_output("squeue -u f0106093 | wc -l",shell=True)
        # njobs must be less than 100
        while int(njobs) > 100:
            time.sleep(6)
            njobs = subprocess.check_output("squeue -u f0106093 | wc -l",shell=True)
        