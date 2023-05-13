import numpy as np
import dill as pickle
import subprocess
import itertools
import os
from glob import glob

data_dir = "traintimes2/optimization"
progress_dir = os.path.join(data_dir, "progress")
log_dir = os.path.join(data_dir, "logfiles")
timelimit_hr = 72
ntasks = 16

def log_interp(points, max_ratio=3.**0.2):
    """
    Makes a log-scale interpolation between the points
    """
    results = []
    
    for v1, v2 in zip(points[:-1], points[1:]):
        results.append(
            np.exp(
                np.arange(np.log(v1), np.log(v2), np.log(max_ratio))
            )
        )
    
    return np.concatenate(results + [[points[-1]]])
    
def get_progress_filename(system, aug_type, pred_type, init_cond, 
        mean_degree, n, train_time):
    os.path.join(progress_dir, 
    'progress-{}-{}-{}-{}-d{}-n{}-tr{:.2f}-vpts.pkl'.format(
        system, aug_type, pred_type, init_cond, 
        mean_degree, n, train_time
    ))
    
def get_results_filename(system, aug_type, pred_type, init_cond, 
        mean_degree, n, train_time):
    os.path.join(data_dir, 
    '{}-{}-{}-{}-d{}-n{}-tr{:.2f}-vpts.pkl'.format(
        system, aug_type, pred_type, init_cond, 
        mean_degree, n, train_time
    ))
    
if __name__=="__main__":
    experiments = list(itertools.product(
        [
        *itertools.product(['lorenz'],log_interp([1.0, 3.0, 10.0, 30.0, 60.0, 100.0])),
        *itertools.product(['thomas'],log_interp([10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0])),
        *itertools.product(['rossler'],log_interp([5.0, 15.0, 50.0, 300.0, 1000.0, 3000.0])),
        ],
        [
            ('augmented', 'activ_f'),
            ('standard', 'activ_f'),
            ('standard', 'random'),
        ],
        [
            'continue',
            'random',
        ],
        [1.0],
        [1000],
    ))
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(progress_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    for ((system, train_time), (aug_type, init_cond), pred_type, mean_degree, n) in experiments:
        experiment = (system, aug_type, pred_type, init_cond, mean_degree, n, train_time)
        # Check if finished previously
        if os.path.exists(get_results_filename(*experiment)):
            continue
            
        # Get the file of parameters we want
        param_filename = f"results/{system}-{aug_type}-{pred_type}-{init_cond}-d{mean_degree}-n{n}.pkl"
        
        args = ('_compare_training_vpts_2.py', param_filename, str(train_time))
            
        flags = (
            '-o', f'{log_dir}/slurm-%j.out',
            '-t', '{}:00:00'.format(timelimit_hr),
            '--ntasks={}'.format(ntasks),
            '--nodes=1',
            '--mem-per-cpu=300M',
            '-J', f'TrT:vpt:{param_filename[:-4]}:{train_time:.2f}',
        )
        
        subprocess.run(['sbatch', *flags, 'parallel_submit.sh', *args])