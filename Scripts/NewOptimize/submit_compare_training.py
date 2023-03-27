
import dill as pickle
import subprocess
import itertools
import os
from glob import glob

data_dir = "traintimes/optimization"
progress_dir = os.path.join(data_dir, "progress")
log_dir = os.path.join(data_dir, "logfiles")
timelimit_hr = 48
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
    
    # Collect the finished jobs
    finished_jobs = set()
    for filename in glob(os.path.join(data_dir, '*.pkl')):
        if filename.endswith('config.pkl'):
            continue
        with open(filename, 'rb') as file:
            # Load and unpack
            experiment, params = pickle.load(file)
            finished_jobs.add(experiment)
    
    for ((system, train_time), (aug_type, init_cond), pred_type, mean_degree, n) in experiments:
        experiment = (system, aug_type, pred_type, init_cond, mean_degree, n, train_time)
        if experiment in finished_jobs:
            continue 
        
        args = ('_compare_training.py', *map(str,experiment))
        flags = (
            '-o', f'{log_dir}/slurm-%j.out',
            '-t', '{}:00:00'.format(timelimit_hr),
            '--ntasks={}'.format(ntasks),
            '--nodes=1',
            '--mem-per-cpu=300M',
            '-J', 'TrT:' + '-'.join(map(str,experiment))
        )
        
        #Submit the job
        subprocess.run(['sbatch', *flags, 'parallel_submit.sh', *args])
        