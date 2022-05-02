import subprocess
import itertools
from glob import glob
import os
import dill as pickle

data_dir = "results"
ntasks = 16
timelimit_hr = 168

systems = [
    'lorenz',
    'rossler',
    'thomas'
]

exp_types = [
    ('augmented', 'activ_f'),
    ('standard', 'activ_f'),
    ('standard', 'random'),
]

pred_types = [
    'continue',
    'random',
]

mean_degrees = [
    0.1,
    1.0,
    2.0
]

n_list = [
    500,
    1000,
    2000
]

def get_completed_jobs():
    """Gets a set of all of the jobs that have completed"""
    finished_jobs = set()
    for filename in glob(os.path.join(data_dir, '*.pkl')):
        if filename.endswith('config.pkl'):
            continue
        with open(filename, 'rb') as file:
            # Load and unpack
            #print(filename)
            experiment, params = pickle.load(file)
            finished_jobs.add(experiment)
    return finished_jobs

subprocess.run(['mkdir', data_dir])
subprocess.run(['mkdir', f"{data_dir}/logfiles"])
subprocess.run(['mkdir', f"{data_dir}/progress"])
progress_dir = os.path.join(data_dir, 'progress')

finished_jobs = get_completed_jobs()
for (system, (aug_type, icmap), pred_type, mean_deg, n) in itertools.product(systems, exp_types, pred_types, mean_degrees, n_list):
    # Check if the experiment is already finished
    if (system, aug_type, pred_type, icmap, mean_deg, n) in finished_jobs:
        continue
    #Get the arguments and flags
    args = ('new_optimize.py', system, aug_type, pred_type, icmap, str(mean_deg), str(n), data_dir, progress_dir)
    flags = (
        '-o', f'{data_dir}/logfiles/slurm-%a.out',
        '-t', '{}:00:00'.format(timelimit_hr),
        '--ntasks={}'.format(ntasks),
        '--nodes=1',
        '--mem-per-cpu=1G',
        '-J', '-'.join(args[1:-1])
    )
    
    #Submit the job
    subprocess.run(['sbatch', *flags, 'parallel_submit.sh', *args])
