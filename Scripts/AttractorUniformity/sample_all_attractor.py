
import numpy as np
import rescomp as rc
from rescomp import optimizer as rcopt
from ipyparallel import Client
import sys


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

def main(sysname, icmap, augtype, rcparams, min_vpt, n_samples, filename, parallel=False, parallel_profile=None, debug=True):
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
    #-Get a decent reservoir computer
    optimizer = rcopt.ResCompOptimizer(sysname, icmap, "random", augtype)
    rescomp = None
    
    if debug:
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
    
    #-Save results to file
    np.savez(filename, train_initial=Utr[0,:], samples=samples, experiment=np.array([sysname, icmap, augtype],dtype=str))

def get_parameters(sysname, icmap, augtype):
    params = {('lorenz', 'activ_f', 'augmented'): {'gamma': 9.887252061575511,
              'mean_degree': 2.0480000000000005,
              'overlap': 0.95,
              'ridge_alpha': 0.01752264528626046,
              'sigma': 0.18286078631906275,
              'spect_rad': 11.48359714330619,
              'window': 1.9545157939863715},
             ('lorenz', 'relax', 'augmented'): {'gamma': 10.259117716857121,
              'mean_degree': 3.154404702761152,
              'overlap': 0.4,
              'ridge_alpha': 0.029562306868021884,
              'sigma': 0.22310987087148082,
              'spect_rad': 4.795034342921333,
              'window': 4.995822961825364},
             ('lorenz', 'relax', 'standard'): {'sigma': 0.01,
              'gamma': 25.0,
              'ridge_alpha': 1e-08,
              'spect_rad': 0.1,
              'mean_degree': 5.0},
             ('lorenz', 'random', 'standard'): {'gamma': 7.671872881713175,
              'mean_degree': 3.55143723946522,
              'ridge_alpha': 0.0013811542757297993,
              'sigma': 0.05275339809393571,
              'spect_rad': 5.487920533874313},
             ('rossler', 'activ_f', 'augmented'): {'gamma': 9.830400000000001,
              'mean_degree': 1.1248983498810199,
              'overlap': 0.95,
              'ridge_alpha': 3.595939840538984e-05,
              'sigma': 0.4372852231511498,
              'spect_rad': 10.019299708108806,
              'window': 8.04691194302645},
             ('rossler', 'relax', 'standard'): {'gamma': 25.0,
              'mean_degree': 2.5694730116992535,
              'ridge_alpha': 3.607695351824033e-05,
              'sigma': 0.1706178229311793,
              'spect_rad': 9.436074416376542},
             ('rossler', 'random', 'standard'): {'gamma': 3.104229610463467,
              'mean_degree': 1.4273198718250377,
              'ridge_alpha': 0.0025686587579037217,
              'sigma': 0.6867776052640227,
              'spect_rad': 7.077730651239169},
             ('thomas', 'activ_f', 'augmented'): {'sigma': 1.1433387142699953,
              'gamma': 12.101591622895805,
              'ridge_alpha': 8.631513359918207e-05,
              'spect_rad': 11.89032219020475,
              'mean_degree': 0.42057901841348283,
              'window': 5.408664556652771,
              'overlap': 0.3},
             ('thomas', 'relax', 'standard'): {'sigma': 1.791187659388025,
              'gamma': 12.874261481466144,
              'ridge_alpha': 3.217960104350012e-05,
              'spect_rad': 8.925945567047068,
              'mean_degree': 2.898050283912454},
             ('thomas', 'random', 'standard'): {'sigma': 2.082780087951669,
              'gamma': 6.492012433078913,
              'ridge_alpha': 1.949576997924246e-05,
              'spect_rad': 7.560213585468283,
              'mean_degree': 0.991321866748073}}
    return params[(sysname, icmap, augtype)]

if __name__ == "__main__":
    if len(sys.argv) >= 7:
        sysname, icmap, augtype = sys.argv[1:4]
        min_vpt = float(sys.argv[4])
        n_samples = int(sys.argv[5])
        filename = sys.argv[6]
        params = get_parameters(sysname, icmap, augtype)
        if len(sys.argv) >= 8:
            parallel=True
            parallel_profile=sys.argv[7]
            if parallel_profile == "None":
                parallel_profile = None
        else:
            parallel=False
            parallel_profile=None
        main(sysname, icmap, augtype, params, min_vpt, n_samples, filename, parallel=parallel, parallel_profile=parallel_profile, debug=True)
    elif len(sys.argv) > 2:
        raise ValueError("Incorrect number of arguments")