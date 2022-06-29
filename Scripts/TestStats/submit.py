import subprocess
import itertools

data_dir = "results/"
n_samples = 500
ntasks = 16

experiments = [
    # Lorenz
    {
        'name': ['lorenz', 'activ_f', 'augmented'],
        'time': 72,
    },
    {
        'name': ['lorenz', 'relax', 'standard'],
        'time': 72,
    },
    {
        'name': ['lorenz', 'random', 'standard'],
        'time': 72,
    },
    # Rossler
    {
        'name': ['rossler', 'activ_f', 'augmented'],
        'time': 72,
    },
    {
        'name': ['rossler', 'relax', 'standard'],
        'time': 72,
    },
    {
        'name': ['rossler', 'random', 'standard'],
        'time': 72,
    },
    # Thomas
    {
        'name': ['thomas', 'activ_f', 'augmented'],
        'time': 72,
    },
    {
        'name': ['thomas', 'relax', 'standard'],
        'time': 72,
    },
    {
        'name': ['thomas', 'random', 'standard'],
        'time': 72,
    },
]


subprocess.run(['mkdir', data_dir])
subprocess.run(['mkdir', f"{data_dir}/logfiles"])

for exp in experiments:
    #Get the arguments and flags
    args = (*exp['name'], str(n_samples), data_dir)
    flags = (
        '-o', '{}/logfiles/slurm-%a.out'.format(data_dir),
        '-t', '{}:00:00'.format(exp['time']),
        '--ntasks={}'.format(ntasks),
        '--nodes=1',
        '--mem-per-cpu=1G',
        '-J', '-'.join(exp['name'])
    )
    
    #Submit the job
    subprocess.run(['sbatch', *flags, 'parallel_submit.sh', *args])