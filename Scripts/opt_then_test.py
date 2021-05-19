#!/bin/python3
import sys
import sherpa
import pickle as pkl
import numpy as np
import rescomp as rc

### Script arguments
# Choose SYSTEM from ["lorenz", "rossler", "thomas", "softrobo"]
# Choose MAP_INITIAL from ["random", "activ_f", "relax"]
# Choose PREDICTION_TYPE from ["continue", "random"]
# Choose METHOD from ["standard", "augmented"]

SYSTEM = sys.argv[1]
MAP_INITIAL = sys.argv[2]
PREDICTION_TYPE = sys.argv[3]
METHOD = sys.argv[4]
EXPERIMENT = (SYSTEM, PREDICTION_TYPE, METHOD)

### Constants
DATADIR = "../Data/" # Contains priors and soft robot data
BIG_ROBO_DATA = "bellows_arm1_whitened.mat"
SMALL_ROBO_DATA = "bellows_arm_whitened.mat"
VPTTOL = 0.5 # Valid prediction time error tolerance
TRAINPER = 0.66 # Percentage of the data used for training
OPT_VPT_REPS = 10
OPT_NTRIALS = 200
NSAVED_ORBITS = 25
LYAP_REPS = 10

# Time Steps for chaotic systems
DT = {
    "lorenz": 0.01,
    "rossler": 0.01,
    "thomas": 0.1
}
# Max orbit time
DURATION = {
    "lorenz": 10,
    "rossler": 150,
    "thomas": 1000
}
# Parameters to optimize in the reservoir computer
RES_OPT_PRMS = [
    "sigma",
    "gamma",
    "ridge_alpha",
    "spect_rad",
    "mean_degree"
]
# Window training algorithm parameters
METHOD_PRMS = [
    "window",
    "overlap"
]
# Additional Soft Robot Parameter
ROBO_OPT_PRMS = [
    "delta"
]

# Default reservoir computer hyper parameters
RES_DEFAULTS = {
    "res_sz":1000,
    "activ_f": lambda x: 1/(1+np.exp(-1*x)),
    "sparse_res":True,
    "uniform_weights":True,
    "signal_dim":3,
    "max_weight":2,
    "min_weight":0,
    "batchsize":2000,
    "map_initial":MAP_INITIAL
}

# Soft robot default parameters
ROBO_DEFAULTS = {
    "signal_dim":6,
    "drive_dim":6
}

### Function definitions

# TODO: Store previous best parameters that can be parsed by sherpa.
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

def loadprior(system):
    """Load best parameters from random searches (Computed previously).
    Parameters:
        system (string): name of the system type being used
    Returns:
        priorprms (List of dictionaries): sets of hyperparameters known to be good"""
    #As far as I can tell, the sherpa function we're giving this to 
    #   wants a list of hyperparameter dictionaries, or a pandas.Dataframe
    #   object of unknown formatting
    with open(f"{system}_prior.pkl", "rb") as file:
        priorprms = pkl.load(file)
        if type(priorprms) is dict:
            #Wrap it in a list
            return [priorprms]
        elif type(priorprms) is list:
            return priorprms
        else:
            print(f"Warning: no correctly-formatted prior data found in {system}_prior.pkl", file=sys.stderr)
    #Return an empty list if we failed to load anything
    return []

def load_robo(filename):
    """Load soft robot order"""
    data = io.loadmat(DATADIR + filename)
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
    slices = ()
    for A in As:
        if axis == 0:
            slices += (A[start:end],)
        if axis == 1:
            slices += (A[:, start:end],)
    return slices

def robo_train_test_split(timesteps=25000, trainper=0.66, test="continue"):
    """Split robot data into training and test chunks """
    t, U, D = load_robo(BIG_ROBO_DATA)
    t, U, D = random_slice(t, U, D, timesteps)
    split_idx = int(np.floor(len(t) * trainper))
    tr, ts = t[:split_idx], t[split_idx:]
    Utr, Uts = U[:split_idx, :], U[split_idx:, :]
    Dtr, Dts = D[:split_idx, :], D[split_idx:, :]
    if test == "random":
        t, U, D = load_robo(SMALL_ROBO_DATA)
        test_timesteps = timesteps - isplit_dx
        ts, Uts, Dts = random_slice(t, U, D, test_timesteps)
    return tr, (Utr, Dtr), (ts, Dts), Uts

def chaos_train_test_split(system, duration=10, trainper=0.66, dt=0.01, test="continue"):
    """ Chaotic system train and test data"""
    tr, Utr, ts, Uts = rc.train_test_orbit(system, duration=duration, trainper=trainper)
    if test == "random":
        test_duration = trainper * duration
        ts, Uts = rc.orbit(system, duration=test_duration, trim=True)
    return tr, Utr, ts, Uts

def train_test_data(system, trainper=0.66, test="continue"):
    """ Load train test data for a given system """
    if system == "softrobo":
        return robo_train_test_split(duration=duration, trainper=trainper, test=test)
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
    err = np.mean( (true - pred)**2 / sig, axis=0)**.5
    return err

def valid_prediction_index(err, tol):
    "First index i where err[i] > tol. err is assumed to be 1D and tol is a float"
    for i in range(len(err)):
        if err[i] > tol:
            return i
    return i

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
    if system == "softrobo":
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
    if system == "softrobo":
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

def build_params(opt_prms):
    """ Extract training method parameters and augment reservoir parameters with defaults.
        Parameters
        ----------
        opt_prms (dict): Dictionary of parameters from the optimizer
    """
    resprms = {}
    methodprms = {}
    for k in opt_prms.keys():
        if k in METHOD_PRMS:
            methodprms[k] = opt_prms[k]
        else:
            resprms[k] = opt_prms[k]
    resprms = {**resprms, **RES_DEFAULTS}
    if SYSTEM == "softrobo":
        resprms = {**ROBO_DEFAULTS, **resprms} # Updates signal_dim and adds drive_dim
    return resprms, methodprms

def vpt(*args, **kwargs):
    """ Compute the valid prediction time for a set of parameters

        Parameters:
        -----------
        system (str): The name of the system from which to generate training data.
            One of: `["lorenz", "rossler", "thomas", "softrobo"]`
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
    err = nrmse(Uts, pre)
    idx = valid_prediction_index(err, VPTTOL)
    vptime = ts[idx-1] - ts[0]
    return vptime

def mean_vpt(*args, **kwargs):
    """ Average valid prediction time across OPT_VPT_REPS repititions """
    tot_vpt = 0
    for i in range(OPT_VPT_REPS):
        tot_vpt += vpt(*args, **kwargs)
    return tot_vpt/OPT_VPT_REPS

def meanlyap(rcomp, r0, ts, pert_size=1e-6):
    """ Average lyapunov exponent across LYAP_REPS repititions """
    lam = 0
    for i in range(LYAP_REPS):
        r0 = init_cond["r0"]
        delta0 = np.random.randn(r0.shape[0]) * pert_size
        predelta = rcomp.predict(ts, r0=r0+delta0)
        i = rc.accduration(pre, predelta)
        lam += rc.lyapunov(ts[:i], pre[:i, :], predelta[:i, :], delta0)
    results["cont_lyap"].append(lam / LYAP_REPS)

### Optimize hyperparameters
parameters = [
    sherpa.Continuous(name='gamma', range=[0.1, 25], ),
    sherpa.Continuous(name='sigma', range=[0.01, 5.0]),
    sherpa.Continuous(name='spect_rad', range=[0.1, 25]),
    sherpa.Continuous(name='ridge_alpha', range=[1e-8, 2], scale='log'),
    sherpa.Continuous(name='mean_degree', range=[0.1, 5]),
]
augmentedprms = [
    sherpa.Continuous(name='window', range=[0.1, 10]),
    sherpa.Discrete(name='overlap', range=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
]
roboprms = [
    sherpa.Continuous(name='delta', range=[0.01, 5.0]),
]
if METHOD == "augmented":
    parameters += augmentedprms
if SYSTEM == "softrobo":
    parameters += roboprms

# Bayesian hyper parameter optimization
priorprms = loadprior(SYSTEM)
algorithm = sherpa.algorithms.GPyOpt(max_num_trials=OPT_NTRIALS, initial_data_points=priorprms)
study = sherpa.Study(parameters=parameters,
                 algorithm=algorithm,
                 lower_is_better=False)

for trial in study:
    exp_vpt = mean_vpt(*EXPERIMENT, **build_params(trial.parameters))
    study.add_observation(trial=trial,
                          objective=exp_vpt)
    study.finalize(trial)
    study.save(DATADIR + SYSTEM) # Need separate directories for each method etc

### Choose the best hyper parameters
# TODO: Figure out how to parse the output of study.get_best_result()
# So that they can be passed into build_params. Basically convert a dataframe
# into a dictionary.
optimized_hyperprms = study.get_best_result()

### Test the training method
results = {name:[] for name in ["continue", "random", "cont_deriv_fit", "rand_deriv_fit", "lyapunov"]}

for k in range(NSAVED_ORBITS):
    tr, Utr, ts, Uts = train_test_data(SYSTEM, trainper=TRAINPER, test="continue")
    resprms, methodprms = build_params(optimized_hyperprms)
    rcomp = trained_rcomp(SYSTEM, tr, Utr, resprms, methodprms)

    ## Continued Prediction
    init_cond = make_initial("continue", rcomp, Uts)
    pre = rcomp_prediction(SYSTEM, rcomp, ts)
    # Compute error and deduce valid prediction time
    err = nrmse(Uts, pre)
    idx = valid_prediction_index(err, VPTTOL)
    vptime = ts[idx-1] - ts[0]
    results["continue"].append(vptime)
    ## Continued Derivative fit
    if SYSTEM != "softrobo":
        err = rc.system_fit_error(ts, pre, SYSTEM)
        trueerr = rc.system_fit_error(ts, Uts, SYSTEM)
        results["cont_deriv_fit"].append((trueerr, err))

    ## Random Prediction
    tr, Utr, ts, Uts = train_test_data(SYSTEM, trainper=TRAINPER, test="random")
    init_cond = make_initial("random", rcomp, Uts)
    pre = rcomp_prediction(SYSTEM, rcomp, ts)
    err = nrmse(Uts, pre)
    idx = valid_prediction_index(err, VPTTOL)
    vptime = ts[idx-1] - ts[0]
    results["random"].append(vptime)
    ## Random Derivative fit
    if SYSTEM != "softrobo":
        err = rc.system_fit_error(ts, pre, SYSTEM)
        trueerr = rc.system_fit_error(ts, Uts, SYSTEM)
        results["rand_deriv_fit"].append((trueerr, err))

    ## Lyapunov Exponent Estimation
    lam = 0
    for i in range(LYAP_REPS):
        r0 = init_cond["r0"]
        delta0 = np.random.randn(DEFAULTS["res_sz"]) * 1e-6
        predelta = rcomp.predict(ts, r0=r0+delta0)
        i = rc.accduration(pre, predelta)
        lam += rc.lyapunov(ts[:i], pre[:i, :], predelta[:i, :], delta0)
    results["cont_lyap"].append(lam / LYAP_REPS)

    # TODO: Save results dictionary with a unique name
    # pkl.dump("unique_name.pkl", results)
