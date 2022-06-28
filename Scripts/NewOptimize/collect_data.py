import pandas as pd
import numpy as np
from glob import glob as glob
import os
import sys
import dill as pickle
from itertools import combinations

def main(source_folder, dest_filename):
    
    df = pd.DataFrame(
        columns=['system','is_aug','pred_type','ic_map','mean_degree','n'])
    for filename in glob(os.path.join(source_folder, '*.pkl')):
        if filename.endswith('config.pkl'):
            continue
        with open(filename, 'rb') as file:
            # Load and unpack
            print(filename)
            experiment, params = pickle.load(file)
            system, is_aug, pred_type, ic_map, mean_degree, n = experiment
            # TODO do stuff with the file
            exp_dictionary = {
                **params,
                'system':system,
                'is_aug':is_aug,
                'pred_type':pred_type,
                'ic_map':ic_map,
                'mean_degree':mean_degree,
                'n':n
            }

            #print(params)
            df = df.concat(exp_dictionary, ignore_index=True)
    
    # Print DataFrame
    print(df)

    # Save DataFrame
    df.to_pickle(dest_filename)
    
        
if __name__=="__main__":
    main(sys.argv[1], sys.argv[2])