import numpy as np
from glob import glob
import dill as pickle
import os
from tqdm import tqdm

def check_has_maxed_time(data_list, thresh, min_n):
    return np.sum(np.array(data_list) >= thresh) >= min_n
    
def main(min_n=100, result_path = 'vpt_results', progress_folder='progress',):
    
    results = set()
    
    max_times = {
        'lorenz':8.0,
        'rossler':120.0,
        'thomas':360.0,
    }
    
    for filename in tqdm(glob(os.path.join(result_path,'*.pkl'))):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            system = data[0][0]
            is_rand = data[0][2]
            threshhold = 0.99 * max_times[system]
            data_list = data[1][is_rand]
            if check_has_maxed_time(data_list, threshhold, min_n):
                results.add(data[0])
    
    print(f"Found {len(results)} experiments with maximum test VPTs.")
    print('\n'.join(map(str,results)))
    print()
    delete = input("Delete these files? [y/n] ")
    
    if delete.lower() == 'y':
        # Delete each of them
        for experiment in tqdm(results):
            res_file = os.path.join(result_path, '{}-{}-{}-{}-d{}-n{}-vpts.pkl'.format(*experiment))
            prog_file = os.path.join(result_path, progress_folder, 'progress-{}-{}-{}-{}-d{}-n{}-vpts.pkl'.format(*experiment)) 
            os.remove(res_file)
            if os.path.exists(prog_file):
                os.remove(prog_file)


if __name__=="__main__":
    main()