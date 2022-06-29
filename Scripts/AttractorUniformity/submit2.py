import subprocess
import itertools


n_samples = 4000
ntasks = 16
data_dir = "AttractorSamples"

experiments = [
    {
        'name':('lorenz', 'activ_f', 'augmented'),
        'time':12,
        'min_vpt':1.7
    },
    {
        'name':('lorenz', 'relax', 'augmented'),
        'time':12,
        'min_vpt':0.25
    },
    {
        'name':('lorenz', 'relax', 'standard'),
        'time':12,
        'min_vpt':0.5
    },
    {
        'name':('lorenz', 'random', 'standard'),
        'time':12,
        'min_vpt':-1
    },
    {
        'name':('rossler', 'activ_f', 'augmented'),
        'time':12,
        'min_vpt':65
    },
    {
        'name':('rossler', 'relax', 'standard'),
        'time':12,
        'min_vpt':15
    },
    {
        'name':('rossler', 'random', 'standard'),
        'time':12,
        'min_vpt':-1
    },
    {
        'name':('thomas', 'activ_f', 'augmented'),
        'time':24,
        'min_vpt':60
    },
    {
        'name':('thomas', 'relax', 'standard'),
        'time':12,
        'min_vpt':60
    },
]

#Ensure the data folder exists
subprocess.run(['mkdir', data_dir])
subprocess.run(['mkdir', f"{data_dir}/logfiles"])

for exp in experiments:
    #Get the arguments, flags and destination file
    target_file = "{}/{}-{}-{}".format(data_dir,*exp['name'])
    args = (*exp['name'], str(exp['min_vpt']), str(n_samples), target_file)
    flags = (
        '-o', '{}/logfiles/slurm-%a.out'.format(data_dir),
        '-t', '{}:00:00'.format(exp['time']),
        '--ntasks={}'.format(ntasks),
        '--nodes=1',
        '--mem-per-cpu=1G',
        '-J', '-'.join(exp['name'])
    )
    #Submit the job
    subprocess.run(['sbatch', *flags, 'parallel_submit2.sh', *args])
