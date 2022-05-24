
import numpy as np
import rescomp as rc
from rescomp import optimizer as rcopt
from ipyparallel import Client
import sys
import dill as pickle


def sample_from_attractor(rescomp, system, n, parallel=False, parallel_profile=None):
    """Returns the VPT from n random-initial-condition predictions of the reservoir computer rc on the given system."""
    
    def _single_sample():
        #Generate signal
        ts, Uts = system.get_random_test()
        init_point = Uts[0,:]
        #Get vpt
        prediction = rescomp.predict(ts, u0=init_point)
        vpt = rcopt.get_vptime(system, ts,Uts, prediction)
        
        return np.array([*init_point, vpt])
        
    if parallel:
        #Initialize parallelization
        client = Client(profile=parallel_profile)
        dview = client[:]
        dview.use_dill()
        dview.block = True
        node_count = len(client.ids)
        dview.execute('import numpy as np')
        dview.execute('from rescomp import optimizer as rcopt')
        
        #Sample the appropriate number of times
        run_ct = int(np.ceil(n / node_count))
        result = dview.apply(lambda k: [_single_sample() for _ in range(k)], run_ct)
        #Unwrap the list properly
        return np.array([item for section in result for item in section])
    else:
        return np.array([_single_sample() for _ in range(n)])

def main(sysname, icmap, augtype, pred_type, rcparams, min_vpt, n_rescomps, n_samples, filename, parallel=False, parallel_profile=None, debug=True):
    """
    Parameters:
        sysname (str) - name of system
        icmap (str) - initial condition mapping 
        augtype (str) - "standard" or "augmented"
        min_vpt (float) - minimum VPT for an accepted reservoir computer. Set to -1 to always accept.
        n (int) - number of samples to take
        filename (str) - where to save results
        parallel_profile (str) - ipyparallel profile name
    """
    results = []
    optimizer = rcopt.ResCompOptimizer(sysname, icmap, "random", augtype)
    # Do modifications to system parameters
    if sysname == 'thomas':
        optimizer.system.dt = 1.0
    elif sysname == 'rossler':
        optimizer.system.dt = 0.125
    
    for N in range(n_rescomps):
        #-Get a decent reservoir computer
        rescomp = None
        
        if debug:
            print(f" -- Iteration {N} -- ")
            print("Generating reservoir computer...")
        while True:
            #Generate a reservoir computer
            (rescomp, tr, Utr, ts, Uts, pre) = optimizer.generate_orbits(1, rcparams, True)[0]
            if rcopt.get_vptime(optimizer.system, ts, Uts, pre) >= min_vpt:
                #Keep this reservoir computer
                break
            else:
                if debug:
                    print("\tRetrying...")
        
        #-Sample a bunch of times
        if debug:
            print("Sampling points")
        samples = sample_from_attractor(rescomp, optimizer.system, n_samples, parallel, parallel_profile)
        
        # Store results
        results.append({
            'train_initial': Utr[0,:],
            'samples': samples,
        })
    
    with open(filename, 'wb') as file:
        pickle.dump({'results':results,
            'experiment':(system, augtype, pred_type, icmap),
            'params':rcparams},
            file)

if __name__ == "__main__":
    # Unwrap command line arguments
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
    

    min_vpt = -1
    n_samples = 8000
    n_rescomps = 8
    filename = "attractor_results" + '/{}-{}-{}-{}-d{}-n{}-attractor.pkl'.format(system, aug_type, pred_type, icmap, mean_deg, n)
    
    if len(sys.argv) >= 14:
        parallel=True
        parallel_profile = sys.argv[13]
        if parallel_profile == "None":
            parallel_profile = None
    else:
        parallel=False
        parallel_profile=None
    main(system, icmap, aug_type, pred_type, params, min_vpt, n_rescomps, n_samples, filename, parallel=parallel, parallel_profile=parallel_profile, debug=True)