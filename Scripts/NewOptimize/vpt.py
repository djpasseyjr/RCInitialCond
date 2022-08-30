import sys
import rescomp as rc
from rescomp import optimizer as rcopt
import pickle
import os

data_dir = 'vpt_results'
progress_dir = os.path.join(data_dir, 'progress')
ntrials = 8096
# break it into pieces that are still divisible by 16
chunk_size = 8096 // 23

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

if __name__=='__main__':
    # system     is_aug pred_type   ic_map  mean_degree     n      gamma   ridge_alpha     sigma  spect_rad  overlap window
    # Get the experiment
    system = sys.argv[1]
    aug_type = sys.argv[2]
    pred_type = sys.argv[3]
    icmap = sys.argv[4]
    mean_deg = float(sys.argv[5])
    n = int(sys.argv[6])

    params = dict()
    params['mean_degree'] = mean_deg
    params['res_sz'] = n
    # Get the other parameters
    params['gamma'] = float(sys.argv[7])
    params['ridge_alpha'] = float(sys.argv[8]) 
    params['sigma'] = float(sys.argv[9]) 
    params['spect_rad'] = float(sys.argv[10]) 
    if aug_type == 'augmented': 
        params['overlap'] = float(sys.argv[11]) 
        params['window'] = float(sys.argv[12]) 

    parallel_profile = sys.argv[-1]

    # Set up and run the jobs
    optimizer = rcopt.ResCompOptimizer(
        system, icmap, pred_type, aug_type,
        parallel_profile=parallel_profile,
        results_directory=data_dir,
    )
    # Modify the dt and test time
    if system == 'lorenz':
        optimizer.system.test_time = 12.0
    elif system == 'thomas':
        pass
    elif system == 'rossler':
        optimizer.system.test_time = 200.0
        
    # Get progress so far, if any
    progress_filename = os.path.join(progress_dir, 'progress-{}-{}-{}-{}-d{}-n{}-vpts.pkl'.format(system, aug_type, pred_type, icmap, mean_deg, n))
    if os.path.exists(progress_filename):
        try:
            with open(progress_filename, 'rb') as file:
                results, n_completed = pickle.load(file)
        except Exception as e:
            print(f"Error reading progress file: {type(e).__class__}: {e}")
            results = None
            n_completed = 0
    else:
        results = None
        n_completed = 0
    
    while n_completed < ntrials:
        next_results = optimizer.run_tests(chunk_size, lyap_reps=1, parameters=params)
        n_completed += chunk_size
        results = merge_dicts(results, next_results)
        # Save progress
        with open(progress_filename, 'wb') as file:
            pickle.dump((results, n_completed), file)
        
    # Save result file
    result_filename = data_dir + '/{}-{}-{}-{}-d{}-n{}-vpts.pkl'.format(system, aug_type, pred_type, icmap, mean_deg, n)
    with open(result_filename, 'wb') as file:
        pickle.dump((
            (system, aug_type, pred_type, icmap, mean_deg, n),
            results
        ), file)
    