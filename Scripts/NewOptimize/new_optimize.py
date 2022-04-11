import sys
import numpy as np
from rescomp import optimizer as rcopt

def main(system, aug_type, pred_type, mean_degree, n):
    """
    aug_type - augmented or standard
    pred_type - continue or random
    """
    mean_degree = float(mean_degree)
    n = int(n)
    system = rcopt.get_system(system)
    optimizer = rcopt.ResCompOptimizer(
        system, 'TODO', pred_type, aug_type,
        rm_params = ['mean_degree'],
        # Rescomp parameters
        mean_degree = mean_degree,
        res_sz = n
    )
    

if __name__=="__main__":
    main(*sys.argv[1:])