import sys
import numpy as np
from rescomp import optimizer as rcopt
import dill as pickle
import os

from submit_compare_training import data_dir, progress_dir

### Global parameters
N_TRIALS = 64
VPT_REPS = 64


def main(system, aug_type, pred_type, init_cond, mean_degree, n, train_time, parallel_profile=None):
    """
    aug_type - augmented or standard
    pred_type - continue or random
    """
    mean_degree = float(mean_degree)
    n = int(n)
    train_time = float(train_time)
    
    optimizer = rcopt.ResCompOptimizer(
        system, init_cond, pred_type, aug_type,
        rm_params = ['mean_degree'],
        parallel_profile=parallel_profile,
        # Intermediate file ## TODO update
        progress_file=os.path.join(progress_dir,'progress-{}-{}-{}-{}-d{}-n{}-tr{}.pkl'.format(system, aug_type, pred_type, init_cond, mean_degree, n, train_time)),
        # Rescomp parameters
        mean_degree = mean_degree,
        res_sz = n,
        ##
    )
    
    optimizer.system.train_time = train_time
    
    # Account for saved ones
    n_trials = N_TRIALS - len(optimizer.opt_observations)
    
    # Run the optimization
    optimizer.run_optimization(
        opt_ntrials=n_trials, 
        vpt_reps=VPT_REPS
    )
    
    # Best params
    best_params = optimizer.get_best_result()
    
    # Save
    result_filename = os.path.join(data_dir, '{}-{}-{}-{}-d{}-n{}-tr{}.pkl'.format(system, aug_type, pred_type, init_cond, mean_degree, n, train_time))
    with open(result_filename, 'wb') as file:
        pickle.dump((
            (system, aug_type, pred_type, init_cond, mean_degree, n, train_time),
            best_params
        ), file)

if __name__=="__main__":
    main(*sys.argv[1:])