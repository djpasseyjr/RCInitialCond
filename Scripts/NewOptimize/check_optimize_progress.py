import itertools
from glob import glob
import os
import dill as pickle

data_dir = "results"
progress_dir = "results/progress"

systems = [
    'lorenz',
    'rossler',
    'thomas'
]

exp_types = [
    ('augmented', 'activ_f'),
    ('standard', 'activ_f'),
    ('standard', 'random'),
]

pred_types = [
    'continue',
    'random',
]

mean_degrees = [
    0.1,
    1.0,
    2.0
]

n_list = [
    500,
    1000,
    2000
]


def get_completed_jobs():
    """Gets a set of all of the jobs that have completed"""
    finished_jobs = set()
    for filename in glob(os.path.join(data_dir, '*.pkl')):
        if filename.endswith('config.pkl'):
            continue
        with open(filename, 'rb') as file:
            # Load and unpack
            #print(filename)
            experiment, params = pickle.load(file)
            finished_jobs.add(experiment)
    return finished_jobs
    
if __name__ == "__main__":
    finished_jobs = get_completed_jobs()
    
    for (system, (aug_type, icmap), pred_type, mean_deg, n) in itertools.product(systems, exp_types, pred_types, mean_degrees, n_list):
        # Check if that job is finished
        if (system, aug_type, pred_type, icmap, mean_deg, n) in finished_jobs:
            continue
        # Otherwise check if there's a progress file
        progress_file = os.path.join(progress_dir,'progress-{}-{}-{}-{}-d{}-n{}.pkl'.format(system, aug_type, pred_type, icmap, mean_deg, n))
        
        print(f'{system} {aug_type} {pred_type} {icmap}: d={mean_deg}, n={n}')
        if os.path.exists(progress_file):
            with open(progress_file, 'rb') as file:
                print(f"\tIterations completed: {len(pickle.load(file))}")
        else:
            print("\tNo progress found")
        print()
        