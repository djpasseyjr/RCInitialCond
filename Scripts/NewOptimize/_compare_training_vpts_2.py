import sys
import dill as pickle
import os
from rescomp import optimizer as rcopt

from submit_ct_vpts_2 import data_dir, get_progress_filename, get_results_filename

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

def main(opt_params, train_time, parallel_profile):
    ntrials = 4096
    chunk_size = 4096 // 16
    
    # Unpack the parameters
    (
        (system, aug_type, pred_type, 
        init_cond, mean_degree, n),
        rc_params
    ) = opt_params
    
    # Set up the optimizer
    optimizer = rcopt.ResCompOptimizer(
        system, init_cond, pred_type, aug_type,
        parallel_profile=parallel_profile,
        parallel=True,
        results_directory=data_dir,
    )
    
    # Modify the test time to be long enough
    if system == 'lorenz':
        optimizer.system.test_time = 12.0
    elif system == 'thomas':
        pass
    elif system == 'rossler':
        optimizer.system.test_time = 200.0
        
    # Set train time
    optimizer.system.train_time = train_time
        
    # Load progress if any so far
    progress_filename = get_progress_filename(system, aug_type, pred_type, init_cond, 
        mean_degree, n, train_time)
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
    result_filename = get_results_filename(system, aug_type, pred_type, 
        init_cond, mean_degree, n, train_time)
    with open(result_filename, 'wb') as file:
        pickle.dump((
            (system, aug_type, pred_type, init_cond, mean_degree, n, train_time),
            results
        ), file)
    
if __name__ == "__main__":
    opt_params_filen = sys.argv[1]
    with open(opt_params_filen,'rb') as file:
        opt_params = pickle.load(file)
    
    train_time = float(sys.argv[2])
    
    if len(sys.argv) > 3:
        parallel_profile = sys.argv[3]
    else:
        parallel_profile = None
    
    main(opt_params, train_time, parallel_profile)