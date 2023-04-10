import sys
import dill as pickle
import os
from rescomp import optimizer as rcopt

from submit_ct_vpts import data_dir, progress_dir

def merge_dicts(first, second):
    """
    Merges two dictionaries of lists.
    """
    if first is None:
        return second
    if second is None:
        return first
        
    # Make copies of items in first
    result = {k:v.copy() for k,v in first.items()}
    # Put second in, appending if needed
    for k,v in second.items():
        if k in first.keys():
            result[k] += v
        else:
            result[k] = v.copy()
    return result
    
def main(opt_params, parallel_profile):
    ntrials = 4096
    chunk_size = 4096 // 8

    # Unpack parameters
    (system, aug_type, pred_type, init_cond, mean_degree, n, train_time), rc_params = opt_params
    
    # Do the vpt stuff
    # Set up and run the jobs
    optimizer = rcopt.ResCompOptimizer(
        system, init_cond, pred_type, aug_type,
        parallel_profile=parallel_profile,
        parallel=True,
        results_directory=data_dir,
    )
    
    # Modify the test time
    if system == 'lorenz':
        optimizer.system.test_time = 12.0
    elif system == 'thomas':
        pass
    elif system == 'rossler':
        optimizer.system.test_time = 200.0
    optimizer.system.train_time = train_time
        
    # Get progress so far, if any
    progress_filename = os.path.join(progress_dir, 'progress-{}-{}-{}-{}-d{}-n{}-tr{}-vpts.pkl'.format(system, aug_type, pred_type, init_cond, mean_degree, n, train_time))
    if os.path.exists(progress_filename):
        try:
            with open(progress_filename, 'rb') as file:
                results, n_completed = pickle.load(file)
        except Exception as e:
            print(f"Error reading progress file:\n\t{type(e).__class__}: {e}")
            results = None
            n_completed = 0
    else:
        results = None
        n_completed = 0
    
    while n_completed < ntrials:
        next_results = optimizer.run_tests(chunk_size, lyap_reps=1, parameters=rc_params)
        n_completed += chunk_size
        results = merge_dicts(results, next_results)
        # Save progress
        with open(progress_filename, 'wb') as file:
            pickle.dump((results, n_completed), file)
        
    # Save result file
    result_filename = os.path.join(data_dir, '{}-{}-{}-{}-d{}-n{}-tr{}-vpts.pkl'.format(system, aug_type, pred_type, init_cond, mean_degree, n, train_time))
    with open(result_filename, 'wb') as file:
        pickle.dump((
            (system, aug_type, pred_type, init_cond, mean_degree, n, train_time),
            results
        ), file)
    

if __name__ == "__main__":
    opt_params_filen = sys.argv[1]
    with open(opt_params_filen,'rb') as file:
        opt_params = pickle.load(file)
    
    parallel_profile = sys.argv[2]
    
    main(opt_params, parallel_profile)