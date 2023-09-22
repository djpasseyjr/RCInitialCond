import sys
import numpy as np
from rescomp import optimizer as rcopt
import dill as pickle
import os

def main(system, window, overlap, aug_type, pred_type, init_cond, mean_degree, n, data_dir, progress_dir, parallel_profile=None):
    """
    aug_type - augmented or standard
    pred_type - continue or random
    """
    mean_degree = float(mean_degree)
    n = int(n)
    window = float(window)
    overlap = float(overlap)
    
    optimizer = rcopt.ResCompOptimizer(
        system, init_cond, pred_type, aug_type,
        rm_params = ['mean_degree', 'window', 'overlap'],
        parallel_profile=parallel_profile, parallel=True,
        # Intermediate file
        progress_file=os.path.join(progress_dir,'progress-{}-{}-{}-{}-d{}-n{}-w{}-o{}.pkl'.format(system, aug_type, pred_type, init_cond, mean_degree, n, window, overlap)),
        # Rescomp parameters
        mean_degree = mean_degree,
        window = window,
        overlap = overlap,
        res_sz = n,
        ##
    )
    
    # Account for saved ones
    n_trials = 50 - len(optimizer.opt_observations)
    
    if n_trials > 0:
        # Run the optimization
        optimizer.run_optimization(
            opt_ntrials=n_trials, 
            vpt_reps=256
        )
    
    # Best params
    n_test = 512
    best_params = optimizer.get_best_result()
    results = optimizer.run_tests(n_test, lyap_reps=1, parameters=best_params)
    
    # Save
    result_filename = os.path.join(data_dir, '{}-{}-{}-{}-d{}-n{}-w{:.4f}-o{:.4f}-vpts.pkl'.format(system, aug_type, pred_type, init_cond, mean_degree, n, window, overlap))
    with open(result_filename, 'wb') as file:
        pickle.dump((
            (system, window, overlap, aug_type, pred_type, init_cond, mean_degree, n),
            best_params, results
        ), file)

if __name__=="__main__":
    main(*sys.argv[1:])
