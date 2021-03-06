#!/bin/python3
"""
Script to perform hyperparameter optimization on a reservoir computer with the specified options.
Run as:
    python3 opt_then_test.py SYSTEM MAP_INITIAL PREDICTION_TYPE METHOD [Results directory] [options...]

### Script arguments ###
Choose SYSTEM from ["lorenz", "rossler", "thomas", "softrobot"]
Choose MAP_INITIAL from ["random", "activ_f", "relax"]
Choose PREDICTION_TYPE from ["continue", "random"]
Choose METHOD from ["standard", "augmented"]

Additional options:
    "--test" - run with testing values.
    "--dashboard" - enable the sherpa dashboard. Not supported on Windows.
    "--parallel=<profile name>" - use parallel processing, on all accessible nodes. Uses the controller with the given profile name. Requires ipyparallel and dill packages to be installed.
"""

import sys
from datetime import datetime

#Check for sufficient arguments before importing anything
if __name__ == "__main__":
    #Extract the additional options from sys.argv
    options = {item for item in sys.argv[1:] if item[:2]=="--"}
    argv = [item for item in sys.argv if item not in options]
    options = {item.split('=')[0]:'='.join(item.split('=')[1:]) for item in options}
    if len(sys.argv) < 5:
        print(__doc__)
        exit()
    SYSTEM = argv[1]
    MAP_INITIAL = argv[2]
    PREDICTION_TYPE = argv[3]
    METHOD = argv[4]
    
    TIMESTAMP = "{:%y%m%d%H%M%S}".format(datetime.now())
    
    if len(argv) > 5:
        results_directory = argv[5]
    else:
        results_directory = None
    
else:
    SYSTEM = None
    MAP_INITIAL = None
    PREDICTION_TYPE = None
    METHOD = None
    options = dict()
    argv = sys.argv
EXPERIMENT = (SYSTEM, PREDICTION_TYPE, METHOD)
PARALLEL = ("--parallel" in options.keys())

import sherpa
import pickle as pkl
import numpy as np
import rescomp as rc
from scipy.io import loadmat
from os import mkdir

    
### Constants
#Load from the relevant .py file
if "--test" in options.keys():
    from parameters.ott_test import *
else:
    from parameters.ott_params import *
    
RES_DEFAULTS["map_initial"] = MAP_INITIAL

if PARALLEL:
    import ipyparallel as ipp
    dview = None
    node_count = 0
    p_profile = options['--parallel']
    
### Function definitions
# These parameters are used as a prior for the bayesian optimization.
# Decent parameters for each chaotic system are stored in rc.SYSTEMS
# A good prior for the softrobot system is:
# ROBO_PRIOR = {
# "res_sz":1000,
# "activ_f": lambda x: 1/(1+np.exp(-1*x)), "sparse_res":True, "uniform_weights":True,
# "signal_dim":6,
# "max_weight":2,
# "min_weight":0,
# "batchsize":2000,
# "drive_dim":6,
# 'delta': 0.3736117214,
# 'gamma': 18.66636932,
# 'mean_degree': 1.7242465519999999, 'ridge_alpha': 1.268554237,
# 'sigma': 0.3125062064,
# 'spect_rad': 0.8922393143999999, 'map_initial': "activ_f"
# }
# Basically change loadprior function to produce the parameters given above
# in a format that sherpa can read

def _set_experiment(*args):
    """
    A helper method to make it easier to set which experiment is being used if this file is imported.
    """
    global SYSTEM, MAP_INITIAL, PREDICTION_TYPE, METHOD, EXPERIMENT
    SYSTEM, MAP_INITIAL, PREDICTION_TYPE, METHOD = args
    EXPERIMENT = (SYSTEM, PREDICTION_TYPE, METHOD)
    RES_DEFAULTS["map_initial"] = MAP_INITIAL
    
def loadprior(system, paramnames):
    """Load best parameters from random searches (Computed previously).
    Parameters not included are set to a default value found in the parameters files.
    Parameters:
        system (string): name of the system type being used
        paramnames (list of strings): name of parameters to keep
    Returns:
        priorprms (List of dictionaries): sets of hyperparameters known to be good"""
    #As far as I can tell, the sherpa function we're giving this to 
    #   wants a list of hyperparameter dictionaries, or a pandas.Dataframe
    #   object of unknown formatting
    def _clean_prior(prior):
        """Removes unneeded parameters and adds all needed parameters"""
        prior = {**PRIOR_DEFAULTS, **prior}
        prior = {key:prior[key] for key in prior if key in paramnames}
        return prior
        
    try:
        with open(DATADIR + f"{system}_prior.pkl", "rb") as file:
            priorprms = pkl.load(file)
            if type(priorprms) is dict:
                #Clean and wrap in a list
                return [_clean_prior(priorprms)]
            elif type(priorprms) is list:
                #Clean each item in the list
                priorprms = [_clean_prior(prms)for prms in priorprms]
                return priorprms
            else:
                print(f"Warning: no correctly-formatted prior data found in {system}_prior.pkl", file=sys.stderr)
    except FileNotFoundError as e:
        print(e)
        print("Using empty prior instead.")
    #Return an empty list if we failed to load anything
    return []

def load_robo(filename):
    """Load soft robot data"""
    data = loadmat(DATADIR + filename)
    t = data['t'][0]
    q = data['q']
    pref = data["pref"]
    return t, q, pref

def random_slice(*args, axis=0):
    """ Take a random slice of an arbitrary number of arrays from the same index
        Parameters:
            As (ndarrays): Arbitrary number of arrays with the same size along the given axis
            slicesize (int): Size of random slice must be smaller than the size of the
                arrays along given axis
        Keyword Parameters:
            axis (int): Axis to slice. Can be 0 or 1.
        Returns
            slices (tuple): A tuple of slices from each array
    """
    As, slicesize = args[:-1], args[-1]
    start = np.random.randint(0, high=len(As[0]) - slicesize + 1)
    end = start + slicesize
    if axis == 0:
        slices = (A[start:end] for A in As)
    if axis == 1:
        slices = (A[:, start:end] for A in As)
    return slices

def robo_train_test_split(timesteps=25000, trainper=0.66, test="continue"):
    """Split robot data into training and test chunks """
    global BIG_ROBO_DATA_LOADED, SMALL_ROBO_DATA_LOADED
    
    if 'BIG_ROBO_DATA_LOADED' not in dir():
        BIG_ROBO_DATA_LOADED = load_robo(BIG_ROBO_DATA)
        SMALL_ROBO_DATA_LOADED = load_robo(SMALL_ROBO_DATA)
    
    t, U, D = BIG_ROBO_DATA_LOADED
    t, U, D = random_slice(t, U, D, timesteps)
    split_idx = int(np.floor(len(t) * trainper))
    tr, ts = t[:split_idx], t[split_idx:]
    Utr, Uts = U[:split_idx, :], U[split_idx:, :]
    Dtr, Dts = D[:split_idx, :], D[split_idx:, :]
    if test == "random":
        t, U, D = SMALL_ROBO_DATA_LOADED
        #Make sure the slice isn't too large
        test_timesteps = int(np.floor(min(timesteps,len(t)) * trainper))
        ts, Uts, Dts = random_slice(t, U, D, test_timesteps)
    return tr, (Utr, Dtr), (ts, Dts), Uts

def chaos_train_test_split(system, duration=10, trainper=0.66, dt=0.01, test="continue"):
    """ Chaotic system train and test data"""
    if test == "random":
        train_duration = trainper * duration
        test_duration = duration - train_duration
        tr, Utr = rc.orbit(system, duration=train_duration, trim=True)
        ts, Uts = rc.orbit(system, duration=test_duration, trim=True)
    else:
        tr, Utr, ts, Uts = rc.train_test_orbit(system, duration=duration, trainper=trainper, dt=dt)
    return tr, Utr, ts, Uts

def train_test_data(system, trainper=0.66, test="continue"):
    """ Load train test data for a given system """
    if system == "softrobot":
        return robo_train_test_split(timesteps=SOFT_ROBO_TIMESTEPS, trainper=trainper, test=test)
    else:
        return chaos_train_test_split(system, duration=DURATION[system], trainper=trainper, dt=DT[system], test=test)

def nrmse(true, pred):
    """ Normalized root mean square error. (A metric for measuring difference in orbits)
    Parameters:
        Two mxn arrays. Axis zero is assumed to be the time axis (i.e. there are m time steps)
    Returns:
        err (ndarray): Error at each time value. 1D array with m entries
    """
    sig = np.std(true, axis=0)
    err = np.linalg.norm((true-pred) / sig, axis=1, ord=2)
    return err

def valid_prediction_index(err, tol):
    "First index i where err[i] > tol. err is assumed to be 1D and tol is a float. If err is never greater than tol, then len(err) is returned."
    mask = err > tol
    if np.any(mask):
        return np.argmax(mask)
    return len(err)

def trained_rcomp(system, tr, Utr, resprms, methodprms):
    """ Returns a reservoir computer trained with the given data and parameters
    Parameters:
        system (str): Name of the system
        tr (ndarray): 1D array of m equally spaced  time values
        Utr (ndarray): mxn 2D array of training signal states
        resparams (dict): Reservoir computer hyperparameters
        methodprms (dict): Training method parameters
    Returns:
        rcomp (ResComp): Trained reservoir computer
    """
    if system == "softrobot":
        rcomp = rc.DrivenResComp(**resprms)
        rcomp.train(tr, *Utr, **methodprms)
    else:
        rcomp = rc.ResComp(**resprms)
        rcomp.train(tr, Utr, **methodprms)
    return rcomp

def rcomp_prediction(system, rcomp, predargs, init_cond):
    """ Make a prediction with the given system
    Parameters:
        system (str): Name of the system to predict
        rcomp (ResComp): Trained reservoir computer
        predargs (variable length arguments): Passed directly into rcomp.predict
        init_cond (dict): Keyword args passed rcomp.predict
    Returns:
        pre (ndarray): Reservoir computer prediction
    """
    if system == "softrobot":
        pre = rcomp.predict(*predargs, **init_cond)
    else:
        pre = rcomp.predict(predargs, **init_cond)
    return pre

def make_initial(pred_type, rcomp, Uts):
    """ Create initial condition for the type of prediction. Either create a reservoir node
        initial condition or use a state space initial condition.
    """
    if pred_type == "continue":
        # Continue evolution of reservoir nodes from current node state
        return {"r0": rcomp.r0}
    else:
        # Use the state space initial condition. (Reservoir will map it to a reservoir node condition)
        return {"u0": Uts[0]}

def build_params(opt_prms, combine=False, system=None):
    """ Extract training method parameters and augment reservoir parameters with defaults.
        Parameters
        ----------
        opt_prms (dict): Dictionary of parameters from the optimizer
        combine (bool): default False; whether to return all parameters as a single dictionary
        system (string): default None; the system to use. If None, takes the value of the global variable SYSTEM
    """
    if system is None:
        system = SYSTEM
        
    if combine:
        if system == "softrobot":
            return {**RES_DEFAULTS, **opt_prms, **ROBO_DEFAULTS}
        else:
            return {**RES_DEFAULTS, **opt_prms}
            
    resprms = {}
    methodprms = {}
    for k in opt_prms.keys():
        if k in METHOD_PRMS:
            methodprms[k] = opt_prms[k]
        else:
            resprms[k] = opt_prms[k]
    resprms = {**RES_DEFAULTS, **resprms}
    if system == "softrobot":
        resprms = {**resprms, **ROBO_DEFAULTS} # Updates signal_dim and adds drive_dim
    return resprms, methodprms

def vpt(*args, **kwargs):
    """ Compute the valid prediction time for a set of parameters

        Parameters:
        -----------
        system (str): The name of the system from which to generate training data.
            One of: `["lorenz", "rossler", "thomas", "softrobot"]`
        pred_type: Predict continuation of training trajectory or predict evolution
            of a random initial condition. One of: `["continue", "random"]`
        method: Training methodology. One of `["standard", "aumented"]`

        The keyword arguments should be parameters from the optimizer (`trial.parameters`),
        parsed by `build_params`.

        Returns:
        -------
        vptime (float): Time in seconds that the reservoir computer was able to predict the
            evolution of the given system with in a fixed tolerance (`VPTOL`) of error.
    """
    system, pred_type, method = args
    # Build train and test data. Soft robot data includes driving signal in Utr and ts.
    tr, Utr, ts, Uts = train_test_data(system, trainper=TRAINPER, test=pred_type)
    # Filter and augment parameters, then build and train a reservoir computer
    resprms, methodprms = build_params(kwargs)
    rcomp = trained_rcomp(system, tr, Utr, resprms, methodprms)
    # Create prediction initial condition and then predict
    init_cond = make_initial(pred_type, rcomp, Uts)
    pre = rcomp_prediction(system, rcomp, ts, init_cond)
    # Compute error and deduce valid prediction time
    vptime = get_vptime(system, ts, Uts, pre)
    return vptime

def mean_vpt(*args, **kwargs):
    """ Average valid prediction time across OPT_VPT_REPS repetitions. Handles parallel processing. """
    if PARALLEL:
        loop_ct = int(np.ceil(OPT_VPT_REPS / node_count))
        vpt_results = dview.apply_sync(lambda ct, *a, **k: [vpt(*a,**k) for _ in range(ct)], loop_ct, *args, **kwargs)
        return np.sum(vpt_results) / (loop_ct * node_count)
    else:
        tot_vpt = 0
        for i in range(OPT_VPT_REPS):
            tot_vpt += vpt(*args, **kwargs)
        return tot_vpt/OPT_VPT_REPS

def get_vptime(system, ts, Uts, pre):
    """
    Valid prediction time for a specific instance
    """
    err = nrmse(Uts, pre)
    idx = valid_prediction_index(err, VPTTOL)
    if idx == 0:
        vptime = 0.
    else:
        if system == "softrobot":
            vptime = ts[0][idx-1] - ts[0][0]
        else:
            vptime = ts[idx-1] - ts[0]
        
    #if "--test" in options.keys():
    #    print(vptime)
    return vptime

def meanlyap(rcomp, pre, r0, ts, pert_size=1e-6, system=None):
    """ Average lyapunov exponent across LYAP_REPS repititions """
    if system is None:
        system = SYSTEM
    
    if system == "softrobot":
        ts, D = ts
    lam = 0
    for i in range(LYAP_REPS):
        delta0 = np.random.randn(r0.shape[0]) * pert_size
        if system == "softrobot":
            predelta = rcomp.predict(ts, D, r0=r0+delta0)
        else:
            predelta = rcomp.predict(ts, r0=r0+delta0)
        i = rc.accduration(pre, predelta)
        lam += rc.lyapunov(ts[:i], pre[:i, :], predelta[:i, :], delta0)
    return lam / LYAP_REPS
    
def test_all(system, optimized_hyperprms):
    """
    Tests a set of optimized hyperparameters for continue and random predictions and derivative fit, as well as Lyapunov exponent.
    Returns, in order:
        Continue vptime
        Random vptime
        Lyapunov exponent
        Continue deriv fit
        Random deriv fit
    The derivative fits will be None if system=='softrobot'.
    """
    results = [None]*5
    
    tr, Utr, ts, Uts = train_test_data(system, trainper=TRAINPER, test="continue")
    resprms, methodprms = build_params(optimized_hyperprms, system=system)
    rcomp = trained_rcomp(system, tr, Utr, resprms, methodprms)
    
    ## Continued Prediction
    init_cond = make_initial("continue", rcomp, Uts)
    pre = rcomp_prediction(system, rcomp, ts, init_cond)
    # Compute error and deduce valid prediction time
    vptime = get_vptime(system, ts, Uts, pre)
    results[0] = vptime
    
    ## Continued Derivative fit
    if system != "softrobot":
        err = rc.system_fit_error(ts, pre, system)
        trueerr = rc.system_fit_error(ts, Uts, system)
        results[3] = (trueerr, err)

    ## Random Prediction
    tr, Utr, ts, Uts = train_test_data(system, trainper=TRAINPER, test="random")
    init_cond = make_initial("random", rcomp, Uts)
    pre = rcomp_prediction(system, rcomp, ts, init_cond)
    vptime = get_vptime(system, ts, Uts, pre)
    results[1] = vptime
    
    ## Random Derivative fit
    if system != "softrobot":
        err = rc.system_fit_error(ts, pre, system)
        trueerr = rc.system_fit_error(ts, Uts, system)
        results[4] = (trueerr, err)
    
    ## Lyapunov Exponent Estimation
    if "r0" in init_cond.keys():
        r0 = init_cond["r0"]
    else:
        if system == "softrobot":
            r0 = rcomp.initial_condition(init_cond["u0"], ts[1][0,:])
        else:
            r0 = rcomp.initial_condition(init_cond["u0"])
    results[2] = meanlyap(rcomp, pre, r0, ts, system=system)
    return tuple(results)

if __name__ == "__main__":
    if "--test" in options.keys():
        print("Running in test mode")
    
    if PARALLEL:
        #Set up things for multithreading
        client = ipp.Client(profile=p_profile)
        dview = client[:]
        dview.use_dill()
        dview.block = True
        node_count = len(client.ids)
        print(f"Using multithreading; running on {node_count} engines.")
        dview.execute('from opt_then_test import *')
        dview.apply(_set_experiment,SYSTEM, MAP_INITIAL, PREDICTION_TYPE, METHOD)
        

    #Find the data directory if none was given as an argument
    if results_directory is None:
        results_directory = "_".join((SYSTEM, MAP_INITIAL, PREDICTION_TYPE, METHOD,TIMESTAMP))
        if "--test" in options.keys():
            results_directory = "TEST-" + results_directory
        results_directory = DATADIR + SYSTEM + "/" + results_directory
    #Make sure the data directory exists
    try:
        mkdir(results_directory)
    except FileExistsError:
        pass
        
    ### Optimize hyperparameters
    param_names = RES_OPT_PRMS
    parameters = [
        sherpa.Continuous(name='gamma', range=[0.1, 25]),
        sherpa.Continuous(name='sigma', range=[0.01, 5.0]),
        sherpa.Continuous(name='spect_rad', range=[0.1, 25]),
        sherpa.Continuous(name='ridge_alpha', range=[1e-8, 2], scale='log'),
        sherpa.Continuous(name='mean_degree', range=[0.1, 5]),
    ]
    augmentedprms = [
        sherpa.Continuous(name='window', range=[10*DT[SYSTEM], 1000*DT[SYSTEM]]),
        sherpa.Ordinal(name='overlap', range=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        #Alternative:
        #sherpa.Continuous(name='overlap', range=[0.0, 0.95])
    ]
    roboprms = [
        sherpa.Continuous(name='delta', range=[0.01, 5.0]),
    ]
    if METHOD == "augmented":
        parameters += augmentedprms
        param_names += METHOD_PRMS
    if SYSTEM == "softrobot":
        parameters += roboprms
        param_names += ROBO_OPT_PRMS
        #Load robot data
        BIG_ROBO_DATA_LOADED = load_robo(BIG_ROBO_DATA)
        SMALL_ROBO_DATA_LOADED = load_robo(SMALL_ROBO_DATA)
        if PARALLEL:
            dview.push({'BIG_ROBO_DATA_LOADED':BIG_ROBO_DATA_LOADED,'SMALL_ROBO_DATA_LOADED':SMALL_ROBO_DATA_LOADED})

    # Bayesian hyper parameter optimization
    priorprms = loadprior(SYSTEM, param_names)
    algorithm = sherpa.algorithms.GPyOpt(max_num_trials=OPT_NTRIALS, initial_data_points=priorprms)
    disable_dashboard = (sys.platform in ['cygwin', 'win32']) or ("--dashboard" not in options)
    study = sherpa.Study(parameters=parameters,
                     algorithm=algorithm,
                     disable_dashboard=disable_dashboard,
                     lower_is_better=False)

    for trial in study:
        try:
            exp_vpt = mean_vpt(*EXPERIMENT, **build_params(trial.parameters, combine=True))
        except Exception as e:
            print("Error encountered.")
            print("Current experiment:", SYSTEM, MAP_INITIAL, PREDICTION_TYPE, METHOD)
            print("Parameter set:", build_params(trial.parameters, combine=True))
            raise e
        study.add_observation(trial=trial,
                              objective=exp_vpt)
        study.finalize(trial)
        study.save(results_directory) # Need separate directories for each method etc

    ### Choose the best hyper parameters
    optimized_hyperprms = study.get_best_result()
    # Trim to only have the actual parameters
    optimized_hyperprms = {key:optimized_hyperprms[key] for key in param_names}

    print("Optimization ran successfully")

    ### Test the training method
    results = {name:[] for name in ["continue", "random", "cont_deriv_fit", "rand_deriv_fit", "lyapunov"]}
    results["experiment"] = (SYSTEM, MAP_INITIAL, PREDICTION_TYPE, METHOD)
    results["opt_parameters"] = optimized_hyperprms
    results["is_test"] = ("--test" in options.keys())

    if PARALLEL:
        #Run test_all() in parallel
        loop_ct = int(np.ceil(NSAVED_ORBITS / node_count))
        test_results = dview.apply_sync(lambda s,p,c:[test_all(s,p) for _ in range(c)], SYSTEM, optimized_hyperprms, loop_ct)
        #Collect the results
        for rlist in test_results:
            for cont_vpt, rand_vpt, lyap, cont_df, rand_df in rlist:
                results["continue"].append(cont_vpt)
                results["random"].append(rand_vpt)
                results["lyapunov"].append(lyap)
                if SYSTEM != 'softrobot':
                    results["cont_deriv_fit"].append(cont_df)
                    results["rand_deriv_fit"].append(rand_df)
    else:
        for k in range(NSAVED_ORBITS):
            cont_vpt, rand_vpt, lyap, cont_df, rand_df = test_all(optimized_hyperprms)
            results["continue"].append(cont_vpt)
            results["random"].append(rand_vpt)
            results["lyapunov"].append(lyap)
            if SYSTEM != 'softrobot':
                results["cont_deriv_fit"].append(cont_df)
                results["rand_deriv_fit"].append(rand_df)
            

    # Save results dictionary with a unique name.
    results_filename = "-".join((SYSTEM, MAP_INITIAL, PREDICTION_TYPE, METHOD, TIMESTAMP)) + ".pkl"
    if "--test" in options.keys():
        results_filename = "TEST-" + results_filename
    with open(results_directory + "/" + results_filename, 'wb') as file:
        pkl.dump(results, file)
    
    print("Testing ran successfully")
    print(f"Results written to {results_directory}/{results_filename}.")
