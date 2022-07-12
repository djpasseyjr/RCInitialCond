
import os
import sys
import subprocess
from glob import glob
import dill as pickle

from submit_compare_training import data_dir as params_dir

data_dir = "traintimes/vpts"
progress_dir = os.path.join(data_dir, "progress")
log_dir = os.path.join(data_dir, "logfiles")
timelimit_hr = 48
ntasks = 16

if __name__ == "__main__":
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(progress_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    for param_filename in glob(os.path.join(params_dir, '*.pkl')):
        with open(param_filename, 'rb') as file:
            (system, aug_type, pred_type, init_cond, mean_degree, n, train_time), best_params = pickle.load(file)
        
        # Check if the corresponding job has finished already
        result_filename = os.path.join(data_dir, '{}-{}-{}-{}-d{}-n{}-tr{}-vpts.pkl'.format(system, aug_type, pred_type, init_cond, mean_degree, n, train_time))
        
        if not os.path.exists(result_filename):
            # Submit the job
            args = ('_compare_training_vpts.py', param_filename)
            
            flags = (
                '-o', f'{log_dir}/slurm-%j.out',
                '-t', '{}:00:00'.format(timelimit_hr),
                '--ntasks={}'.format(ntasks),
                '--nodes=1',
                '--mem-per-cpu=300M',
                '-J', 'TrT:vpt:'+param_filename[:-4],
            )
            
            subprocess.run(['sbatch', *flags, 'parallel_submit.sh', *args])
            