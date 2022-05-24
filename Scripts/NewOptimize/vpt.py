import sys
import rescomp as rc
from rescomp import optimizer as rcopt
import pickle

data_dir = 'vpt_results'
ntrials = 8092

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
    # Modify the dt
    if system == 'thomas':
        optimizer.system.dt = 1.0
    elif system == 'rossler':
        optimizer.system.dt = 0.125

    results = optimizer.run_tests(ntrials, lyap_reps=1, parameters=params)

    # Save result file
    result_filename = data_dir + '/{}-{}-{}-{}-d{}-n{}-vpts.pkl'.format(system, aug_type, pred_type, icmap, mean_deg, n)
    with open(result_filename, 'wb') as file:
        pickle.dump((
            (system, aug_type, pred_type, icmap, mean_deg, n),
            results
        ), file)
    