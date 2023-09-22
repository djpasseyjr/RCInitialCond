import subprocess
import itertools
from glob import glob
import os
import dill as pickle
import numpy as np

# Submission script to test different window sizes and overlaps #

data_dir = "results_grid_search"
ntasks = 16
timelimit_hr = 72


def logspace(x1, x2, n_points):
    return np.exp(np.linspace(np.log(x1), np.log(x2), n_points))
    
# Different window sizes #
window_sizes_lorenz = logspace(0.02, 6.6, 30)
window_sizes_rossler = logspace(0.1, 165, 30)
window_sizes_thomas = logspace(1, 660, 30)

overlaps = np.linspace(0,0.95,25)

systems_windows_overlaps = [
    *itertools.product(['lorenz'],window_sizes_lorenz,overlaps),
    *itertools.product(['rossler'],window_sizes_rossler,overlaps),
    *itertools.product(['thomas'],window_sizes_thomas,overlaps)
]

exp_types = [
    ('augmented', 'activ_f'),
]

pred_types = [
    'random'
]

mean_degrees = [
#    0.1,
    1.0,
#    2.0,
]

n_list = [
#    500,
    1000,
#    2000,
]

if __name__ == "__main__":
    subprocess.run(['mkdir', '-p', data_dir])
    subprocess.run(['mkdir', '-p', f"{data_dir}/logfiles"])
    subprocess.run(['mkdir', '-p', f"{data_dir}/progress"])
    progress_dir = os.path.join(data_dir, 'progress')

    for ((system, window, overlap), (aug_type, icmap), pred_type, mean_deg, n) in itertools.product(systems_windows_overlaps, exp_types, pred_types, mean_degrees, n_list):
        # Check if the experiment is already finished
        if os.path.exists(
            os.path.join(data_dir, '{}-{}-{}-{}-d{}-n{}-w{:.4f}-o{:.4f}-vpts.pkl'.format(system, aug_type, pred_type, icmap, mean_deg, n, window, overlap))
        ):
            continue
            
        #Get the arguments and flags
        args = ('window_grid_optimize.py', system, str(window), str(overlap), aug_type, pred_type, icmap, str(mean_deg), str(n), data_dir, progress_dir)
        flags = (
            '-o', f'{data_dir}/logfiles/slurm-%j.out',
            '-t', '{}:00:00'.format(timelimit_hr),
            '--ntasks={}'.format(ntasks),
            '--nodes=1',
            '--mem-per-cpu=800M',
            '-J', '-'.join(args[1:-1])
        )

        #Submit the job
        subprocess.run(['sbatch', *flags, 'parallel_submit.sh', *args])
