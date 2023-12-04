import numpy as np
import dill as pickle
import subprocess
import itertools
import os
from glob import glob

data_dir = "errvpts/"
progress_dir = os.path.join(data_dir, "progress")
log_dir = os.path.join(data_dir, "logfiles")
timelimit_hr = 72
ntasks = 16

def get_progress_filename(system, aug_type, pred_type, init_cond, 
        mean_degree, n, err_amt):
    return os.path.join(progress_dir, 
        'progress-{}-{}-{}-{}-d{}-n{}-err{:.2f}-vpts.pkl'.format(
            system, aug_type, pred_type, init_cond, 
            mean_degree, n, err_amt
        )
    )
    
def get_results_filename(system, aug_type, pred_type, init_cond, 
        mean_degree, n, err_amt):
    return os.path.join(data_dir, 
        '{}-{}-{}-{}-d{}-n{}-err{:.2f}-vpts.pkl'.format(
            system, aug_type, pred_type, init_cond, 
            mean_degree, n, err_amt
        )
    )
    
if __name__=="__main__":
    experiments = list(itertools.product(
        [
        *itertools.product(['lorenz'],[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]),
        *itertools.product(['thomas'],[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]),
        *itertools.product(['rossler'],[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]),
        ],
        [ # Training method
            ('augmented', 'activ_f'),
        #    ('standard', 'activ_f'),
        #    ('standard', 'random'),
        ],
        [ # Prediction type
        #    'continue',
            'random',
        ],
        [1.0],
        [1000],
    ))
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    for ((system, err_amt), (aug_type, init_cond), pred_type, mean_degree, n) in experiments:
        experiment = (system, aug_type, pred_type, init_cond, mean_degree, n, err_amt)
        # Check if finished previously
        if os.path.exists(get_results_filename(*experiment)):
            continue
            
        # Get the file of parameters we want
        param_filename = f"results/{system}-{aug_type}-{pred_type}-{init_cond}-d{mean_degree}-n{n}.pkl"
        
        args = ('_err_vpts.py', param_filename, str(err_amt))
            
        flags = (
            '-o', f'{log_dir}/slurm-%j.out',
            '-t', '{}:00:00'.format(timelimit_hr),
            '--ntasks={}'.format(ntasks),
            '--nodes=1',
            '--mem-per-cpu=300M',
            '-J', f'TrT:vpt:{err_amt:.2f}:{param_filename[:-4]}',
        )
        
        subprocess.run(['sbatch', *flags, 'parallel_submit.sh', *args])