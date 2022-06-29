import subprocess
from datetime import datetime

EXPERIMENTS = [[('softrobot', 'activ_f', 'continue', 'standard'),   72],
                [('softrobot', 'relax', 'continue', 'standard'),    72]]

subdir = "results_{:%y%m%d%H%M%S}".format(datetime.now())
subprocess.run(['mkdir', subdir])

for experiment, time in EXPERIMENTS:
    log_dir = f"{subdir}/{'_'.join(experiment)}"
    subprocess.run(['mkdir', log_dir])
    flags=('-o', f"{log_dir}/slurm-%a.out", '-t', f"{time}:00:00", "--ntasks=16", "--nodes=1", "--mem-per-cpu=2G", '-J', '-'.join(experiment))
    subprocess.run(['sbatch', *flags, 'parallel_submit.sh', *experiment, log_dir])