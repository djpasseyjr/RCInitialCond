import numpy as np
import sys
import dill as pickle
import os
from rescomp import optimizer as rcopt

from submit_err_vpts import data_dir, get_progress_filename, get_results_filename

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
    
def bind_function(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the 
    instance as the first argument, i.e. "self".
    
    Source: https://stackoverflow.com/questions/1015307/python-bind-an-unbound-method#comment8431145_1015405
    """
    if as_name is None:
        as_name = func.__name__
    #Get the function as though it were a method of instance (which it then is)
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method

def main(opt_params, err_amt, parallel_profile):
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
        parallel=(parallel_profile is not None),
        results_directory=data_dir,
    )
    
    # Modify the test time to be long enough
    if system == 'lorenz':
        optimizer.system.test_time = 12.0
    elif system == 'thomas':
        pass
    elif system == 'rossler':
        optimizer.system.test_time = 200.0
        
    # Modify the data-generating method to add in noise into training data
    real_get_train_test_data = optimizer.system.get_train_test_data
    
    def replaced_get_train_test_data(self, *args, **kwargs):
        import numpy as np
        tr, Utr, ts, Uts = real_get_train_test_data(*args, **kwargs)
        Utr += np.random.normal(scale=err_amt, size=Utr.shape)
        return tr, Utr, ts, Uts
        
    bind_function(optimizer.system, replaced_get_train_test_data, as_name="get_train_test_data")
        
    # Load progress if any so far
    progress_filename = get_progress_filename(system, aug_type, pred_type, init_cond, 
        mean_degree, n, err_amt)
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
        init_cond, mean_degree, n, err_amt)
    with open(result_filename, 'wb') as file:
        pickle.dump((
            (system, aug_type, pred_type, init_cond, mean_degree, n, err_amt),
            results
        ), file)
    
if __name__ == "__main__":
    opt_params_filen = sys.argv[1]
    with open(opt_params_filen,'rb') as file:
        opt_params = pickle.load(file)
    
    err_amt = float(sys.argv[2])
    
    if len(sys.argv) > 3:
        parallel_profile = sys.argv[3]
    else:
        parallel_profile = None
    
    main(opt_params, err_amt, parallel_profile)