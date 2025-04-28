
import yaml
import time 
import subprocess

from jobs.run_sweep import start_sweep, submit_sweep_jobs

def create_yaml_config(sweep_name, constant_param, param_value):
    '''
    write a yaml file with the sweep configuration
    '''
    if constant_param == 'p':
        config = {
            'program': 'src/sweep.py',
            'name': sweep_name,
            'method': 'random',
            'metric': {
                'name': 'f1',
                'goal': 'maximize'
            },
            'parameters': {
                'p': {
                    'values': [max(param_value-0.5,0.1),param_value,param_value+0.5]
                },
                'q': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 10
                },
                'g': {
                    'values': [0,1,2]
                }
            },
            'command': [
                'python',
                '${program}',
                '--n2v',
                'OTF',
                '--sweep',
                sweep_name,
                '--seed',
                '30',
                '${args}',
            ]
        }
    elif constant_param == 'q':
        config = {
            'program': 'src/sweep.py',
            'name': 'stability_sweep',
            'method': 'random',
            'metric': {
                'name': 'f1',
                'goal': 'maximize'
            },
            'parameters': {
                'q': {
                    'values': [max(param_value-0.5,0.1),max(param_value-0.1,0.1),param_value,param_value+0.1,param_value+0.5]
                },
                'p': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 20
                },
                'g': {
                    'values': [0,1,2]
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
    file_name = f'configs/sweep_config_{sweep_name}.yaml'
    with open(file_name, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    return file_name


if __name__ == '__main__':
    
    param_dict = { 'p' : [19,1,23,14,18,100,7,15,5,9,37.88760595314121,2,0.01,50,18.76305677654519,6],
                'q' : [8.483911078685804, 0.1, 9.58221644861958, 8.017332462471247,9.02662322131865, 
                            100, 3.146340793125213,0.946419088343395, 7.0656378447109605, 3.901922405213064,
                            7.078691990493878, 1.6648497527490949, 5.219434543005656, 9.695352977175908, 
                            3.818854151367598, 0.01, 1, 2.550430386372319, 4.408336461091874]}
    sweep_name = 'stability_sweep'

    for param, vals in param_dict.items():
            for val in vals:
                file_name = create_yaml_config(sweep_name, param, val)
                sweep_id = start_sweep(file_name, sweep_name)
                submit_sweep_jobs(sweep_id, sweep_name, 50)
                njobs = subprocess.check_output("squeue -u f0106093 | wc -l",shell=True)
                # njobs must be less than max jobs allowed
                while int(njobs) > 100:
                    time.sleep(6)
                    njobs = subprocess.check_output("squeue -u f0106093 | wc -l",shell=True)

