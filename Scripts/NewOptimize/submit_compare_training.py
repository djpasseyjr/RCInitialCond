
import subprocess
import itertools
import os
from glob import glob

data_dir = "traintimes/optimization"
progress_dir = os.path.join(data_dir, "progress")
log_dir = os.path.join(data_dir, "logfiles")
timelimit_hr = 48
ntasks = 16

if __name__=="__main__":
    experiments = list(itertools.product(
        [
        *itertools.product(['lorenz'],[1.0, 3.0, 10.0, 30.0]),
        *itertools.product(['thomas'],[10.0, 100.0, 1000.0]),
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
        print('sbatch', *flags, 'parallel_submit.sh', *args)
        #subprocess.run(['sbatch', *flags, 'parallel_submit.sh', *args])
        