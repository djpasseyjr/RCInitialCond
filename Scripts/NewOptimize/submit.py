import subprocess
import itertools

data_dir = "results"
ntasks = 16
timelimit_hr = 72

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


subprocess.run(['mkdir', data_dir])
subprocess.run(['mkdir', f"{data_dir}/logfiles"])

for (system, (aug_type, icmap), pred_type, mean_deg, n) in itertools.product(systems, exp_types, pred_types, mean_degrees, n_list):
    #Get the arguments and flags
    args = ('new_optimize.py', system, aug_type, pred_type, icmap, str(mean_deg), str(n), data_dir)
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