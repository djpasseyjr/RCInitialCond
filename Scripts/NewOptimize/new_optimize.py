import sys
import numpy as np
from rescomp import optimizer as rcopt
import dill as pickle

def main(system, aug_type, pred_type, init_cond, mean_degree, n, parallel_profile=None):
    """
    aug_type - augmented or standard
    pred_type - continue or random
    """
    mean_degree = float(mean_degree)
    n = int(n)
    
    optimizer = rcopt.ResCompOptimizer(
        system, init_cond, pred_type, aug_type,
        rm_params = ['mean_degree'],
        parallel_profile=parallel_profile,
        # Rescomp parameters
        mean_degree = mean_degree,
        res_sz = n,
        ##
    )
    
    # Run the optimization
    optimizer.run_optimization(
        opt_ntrials=100, 
        vpt_reps=256
    )
    
    # Best params
    best_params = optimizer.get_best_result()
    
    # Save
    with open(result_file, 'wb') as file:
        pickle.dump((
            (system, aug_type, pred_type, init_cond, mean_degree, n),
            best_params
        ), file)

if __name__=="__main__":
    main(*sys.argv[1:])