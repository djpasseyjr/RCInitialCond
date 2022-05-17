import sys
import numpy as np
import rescomp as rc
import pandas as pd
import subprocess
from glob import glob
import os
import dill as pickle

if __name__ == '__main__':
    data_dir = "attractor_results"
    ntasks = 16
    timelimit_hr = 48

    subprocess.run(['mkdir', data_dir])
    subprocess.run(['mkdir', f"{data_dir}/logfiles"])

    # Get the ones that have finished
    finished_jobs = set()
    for filename in glob(os.path.join(data_dir, '*.pkl')):
        if filename.endswith('config.pkl'):
            continue
        with open(filename, 'rb') as file:
            # Load and unpack
            #print(filename)
            experiment, result = pickle.load(file)
            finished_jobs.add(experiment)

    # Load dataframe
    df = pd.read_pickle(sys.argv[1])

    for x in df.values:
        # Check if that one's been finished
        if (x[0], x[1], x[2], x[3], x[4], x[5]) in finished_jobs:
            continue
                
        args = (
            'sample_all_attractor.py', 
            x[0], # system
            x[1], # aug_type
            x[2], # pred_type
            x[3], # icmap
            str(x[4]), # mean_deg
            str(x[5]), # n
            str(x[6]), # gamma
            str(x[7]), # ridge_alpha
            str(x[8]), # sigma
            str(x[9]), # spect_rad
            str(x[10]), # overlap
            str(x[11]) # window
        )

        flags = (
            '-o', f'{data_dir}/logfiles/slurm-%a.out',
            '-t', '{}:00:00'.format(timelimit_hr),
            '--ntasks={}'.format(ntasks),
            '--nodes=1',
            '--mem-per-cpu=1G',
            '-J', 'atr-' + '-'.join(args[1:7])
        )

        subprocess.run(['sbatch', *flags, 'parallel_submit.sh', *args])