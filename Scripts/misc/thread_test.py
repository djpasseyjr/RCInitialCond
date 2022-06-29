import subprocess
from datetime import datetime

subdir = "parallel_test"
subprocess.run(['mkdir', f"../Data/{subdir}"])
experiment = ('rossler', 'random', 'random', 'augmented')
timestamp = "{:%y%m%d%H%M%S}".format(datetime.now())
log_dir = f"../Data/{subdir}/{'_'.join((*experiment, timestamp))}"
subprocess.run(['mkdir', log_dir])
flags=('-o', f"{log_dir}/slurm-%a.out", '-t', f"00:20:00", "--ntasks=5", "--nodes=1", "--mem-per-cpu=1024M", '-J', 'thread-test')
subprocess.run(['sbatch', *flags, 'parallel_submit.sh', *experiment, log_dir])