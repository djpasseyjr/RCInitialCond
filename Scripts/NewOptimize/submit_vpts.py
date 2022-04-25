import sys
import numpy as np
import rescomp as rc
import pandas as pd
import subprocess

if __name__ == '__main__':
    data_dir = "vpt_results"
    ntasks = 16
    timelimit_hr = 72

    subprocess.run(['mkdir', data_dir])
    subprocess.run(['mkdir', f"{data_dir}/logfiles"])

    # Load dataframe
    df = pd.read_pickle(sys.argv[1])

    for x in df.values:
        args = (
            'vpt.py', 
            x[0], 
            x[1], 
            x[2], 
            x[3], 
            str(x[4]),
            str(x[5]),
            str(x[6]),
            str(x[7]),
            str(x[8]),
            str(x[9]),
            str(x[10]),
            str(x[11])
        )

        flags = (
            '-o', f'{data_dir}/logfiles/slurm-%a.out',
            '-t', '{}:00:00'.format(timelimit_hr),
            '--ntasks={}'.format(ntasks),
            '--nodes=1',
            '--mem-per-cpu=1G',
            '-J', '-'.join(args[1:7])
        )

        subprocess.run(['sbatch', *flags, 'parallel_submit.sh', *args])