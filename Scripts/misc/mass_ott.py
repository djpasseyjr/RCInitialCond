import subprocess
import itertools
from datetime import datetime

EXPERIMENTS = [
                #[('softrobot', 'activ_f', 'continue', 'standard'),   48], batch 3
                #[('softrobot', 'activ_f', 'random', 'standard'),    72], batch 5
                #[('softrobot', 'random', 'continue', 'augmented'),  72], batch 5
                #[('softrobot', 'relax', 'continue', 'augmented'),   72], batch 5
                #[('softrobot', 'random', 'random', 'augmented'),    48], batch 3
                #[('softrobot', 'activ_f', 'continue', 'augmented'), 72], batch 5
                #[('softrobot', 'relax', 'random', 'standard'),      48], batch 3
                #[('softrobot', 'random', 'continue', 'standard'),   48], batch 3
                #[('softrobot', 'activ_f', 'random', 'augmented'),   48], batch 3
                #[('softrobot', 'random', 'random', 'standard'),     48], batch 3
                #[('softrobot', 'relax', 'continue', 'standard'),    48], batch 3
                #[('softrobot', 'relax', 'random', 'augmented'),     48], batch 3
                #[('thomas', 'relax', 'continue', 'augmented'),      72], batch 5
                #[('thomas', 'random', 'random', 'augmented'),       48], batch 4
                #[('thomas', 'activ_f', 'random', 'augmented'),      72], batch 5
                #[('thomas', 'relax', 'random', 'augmented'),        72], batch 5
                #[('thomas', 'random', 'continue', 'augmented'),     24], batch 3
                #[('thomas', 'activ_f', 'continue', 'augmented'),    13], batch 2
                #[('rossler', 'relax', 'continue', 'augmented'),     72], batch 5
                #[('rossler', 'random', 'random', 'augmented'),      24], batch 3
                #[('rossler', 'relax', 'random', 'augmented'),       72], batch 5
                [('rossler', 'activ_f', 'random', 'augmented'),     24],
                [('rossler', 'relax', 'continue', 'standard'),     24],
                [('rossler', 'relax', 'random', 'standard'),     24],
                [('rossler', 'random', 'continue', 'standard'), 24],
                [('rossler', 'random', 'continue', 'augmented'), 24],
                [('rossler', 'activ_f', 'continue', 'standard'), 24],
                [('rossler', 'activ_f', 'continue', 'augmented'), 24],
                [('lorenz', 'random', 'continue', 'augmented'), 16],
                [('lorenz', 'random', 'random', 'augmented'), 16],
                [('lorenz', 'random', 'continue', 'standard'), 16],
                [('lorenz', 'activ_f', 'continue', 'augmented'), 16],
                [('lorenz', 'activ_f', 'random', 'augmented'), 16],
                [('lorenz', 'activ_f', 'continue', 'standard'), 16],
                [('lorenz', 'relax', 'continue', 'augmented'), 16],
                [('lorenz', 'relax', 'continue', 'standard'), 16],
            ]
                
#EXPERIMENTS = [(('rossler', *item), 24) for item in itertools.product(['relax','random','activ_f'],['continue','random'],['standard','augmented'])]

subdir = "results8"
subprocess.run(['mkdir', f"../Data/{subdir}"])
for experiment, time in EXPERIMENTS:
    timestamp = "{:%y%m%d%H%M%S}".format(datetime.now())
    log_dir = f"../Data/{subdir}/{'_'.join((*experiment, timestamp))}"
    subprocess.run(['mkdir', log_dir])
    flags=('-o', f"{log_dir}/slurm-%a.out", '-t', f"{time}:00:00", "--ntasks=16", "--nodes=1", "--mem-per-cpu=2G", '-J', '-'.join(experiment))
    subprocess.run(['sbatch', *flags, 'parallel_submit.sh', *experiment, log_dir])