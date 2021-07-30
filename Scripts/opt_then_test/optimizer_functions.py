import rescomp as rc
import pickle as pkl
import numpy as np

from optimizer_systems import get_prior_defaults, get_res_defaults

from sherpa import Continuous, Ordinal

#########################
## Testing functions
#########################

def vpt(*args, **kwargs):
    """ Compute the valid prediction time for a set of parameters

        Parameters:
        -----------
        system (str): The name of the system from which to generate training data.
            One of: `["lorenz", "rossler", "thomas", "softrobot"]`
        pred_type: Predict continuation of training trajectory or predict evolution
            of a random initial condition. One of: `["continue", "random"]`

        The keyword arguments should be parameters from the optimizer (`trial.parameters`),
        parsed by `build_params`.

        Returns:
        -------
        vptime (float): Time in seconds that the reservoir computer was able to predict the
            evolution of the given system with in a fixed tolerance (`VPTOL`) of error.
    """
    system, pred_type = args
    
    tr, Utr, ts, Uts, pre = create_orbit(*args, **kwargs)
    
    # Compute error and deduce valid prediction time
    vptime = get_vptime(system, ts, Uts, pre)
    return vptime
    
def test_all(system, lyap_reps=20, **opt_params):
    """
    Tests a set of hyperparameters for continue and random predictions and derivative fit, as well as Lyapunov exponent.
    Returns, in order:
        Continue vptime
        Random vptime
        Lyapunov exponent
        Continue deriv fit
        Random deriv fit
    The derivative fits will be None if system.is_diffeq is False.
    """
    results = [None]*5
    
    #Get the training and continue test data
    tr, Utr, tcs, Ucs = system.get_train_test_data(True)
    rcomp = trained_rcomp(system, tr, Utr, **opt_params)
    
    ## Continued Prediction
    init_cond = make_initial("continue", rcomp, Ucs)
    pre = rcomp_prediction(system, rcomp, tcs, init_cond)
    # Compute error and deduce valid prediction time
    vptime = get_vptime(system, tcs, Ucs, pre)
    results[0] = vptime
    
    ## Continued Derivative fit
    if system.is_diffeq:
        err = system.system_fit_error(tcs, pre)
        trueerr = system.system_fit_error(tcs, Ucs)
        results[3] = (trueerr, err)

    ## Random Prediction
    trs, Urs = system.get_random_test()
    init_cond = make_initial("random", rcomp, Urs)
    pre = rcomp_prediction(system, rcomp, trs, init_cond)
    vptime = get_vptime(system, trs, Urs, pre)
    results[1] = vptime
    
    ## Random Derivative fit
    if system.is_diffeq:
        err = system.system_fit_error(trs, pre)
        trueerr = system.system_fit_error(trs, Urs)
        results[4] = (trueerr, err)
    
    ## Lyapunov Exponent Estimation
    if "r0" in init_cond.keys():
        r0 = init_cond["r0"]
    else:
        if system.is_driven:
            r0 = rcomp.initial_condition(init_cond["u0"], trs[1][0,:])
        else:
            r0 = rcomp.initial_condition(init_cond["u0"])
    results[2] = meanlyap(system, rcomp, pre, r0, trs, lyap_reps=lyap_reps)
    return tuple(results)
    
def meanlyap(system, rcomp, pre, r0, ts, pert_size=1e-6, lyap_reps=20):
    """ Average lyapunov exponent across lyap_reps repititions """
    if system.is_driven:
        ts, D = ts
    lam = 0
    for i in range(lyap_reps):
        delta0 = np.random.randn(r0.shape[0]) * pert_size
        if system.is_driven:
            predelta = rcomp.predict(ts, D, r0=r0+delta0)
        else:
            predelta = rcomp.predict(ts, r0=r0+delta0)
        i = rc.accduration(pre, predelta)
        lam += rc.lyapunov(ts[:i], pre[:i, :], predelta[:i, :], delta0)
    return lam / lyap_reps
    
def create_orbit(*args, **kwargs):
    """ 
    Trains a reservoir computer and has it predict, using the given arguments
    
        Parameters:
        -----------
        system (str): The name of the system from which to generate training data.
            One of: `["lorenz", "rossler", "thomas", "softrobot"]`
        pred_type: Predict continuation of training trajectory or predict evolution
            of a random initial condition. One of: `["continue", "random"]`

        The keyword arguments should be parameters from the optimizer (`trial.parameters`),
        parsed by `build_params`.

        Returns:
        -------
        tr, Utr: time and system state used for training
        ts, Uts: time and system state used for testing
        pre: reservoir computer prediction at the time points in 'ts'
    """
    system, pred_type = args
    #Get the training and test data
    tr, Utr, ts, Uts = system.get_train_test_data(pred_type=='continue')
    # Build and train a reservoir computer
    rcomp = trained_rcomp(system, tr, Utr, **kwargs)
    # Create prediction initial condition and then predict
    init_cond = make_initial(pred_type, rcomp, Uts)
    pre = rcomp_prediction(system, rcomp, ts, init_cond)
    
    return tr, Utr, ts, Uts, pre

#########################
## Rescomp creation
#########################

def make_initial(pred_type, rcomp, U):
    """ Create initial condition for the type of prediction. Either create a reservoir node
        initial condition or use a state space initial condition.
    """
    if pred_type == "continue":
        # Continue evolution of reservoir nodes from current node state
        return {"r0": rcomp.r0}
    else:
        # Use the state space initial condition. (Reservoir will map it to a reservoir node condition)
        return {"u0": U[0]}
  
def trained_rcomp(system, tr, Utr, res_ode=None, **opt_params):
    """ Returns a reservoir computer trained with the given data and parameters
    Parameters:
        system (template.System): The system being used to train
        tr (ndarray): 1D array of m equally spaced  time values
        Utr (ndarray): mxn 2D array of training signal states. If driven, actually tuple of (Utr, Dtr)
        res_ode (dict->callable, optional): if specified, replaces the reservoir computer's internal ODEs before training
        opt_params: Parameters to be used in creating and training the reservoir computer
    Returns:
        rcomp (ResComp): Trained reservoir computer
    """
    resprms, methodprms, otherprms = build_params(system, opt_params)
    if system.is_driven:
        rcomp = rc.DrivenResComp(**resprms)
    else:
        rcomp = rc.ResComp(**resprms)
    
    if res_ode is not None:
        #ResComp and DrivenResComp call the ODE functions different things
        if system.is_driven:
            bind_function(rcomp, res_ode['res_ode'], 'res_f')
            bind_function(rcomp, res_ode['trained_res_ode'], 'res_pred_f')
        else:
            bind_function(rcomp, res_ode['res_ode'], 'res_ode')
            bind_function(rcomp, res_ode['trained_res_ode'], 'trained_res_ode')
        initial = res_ode.get('initial_condition')
        if initial is not None:
            bind_function(rcomp, initial, 'initial_condition')
            
    for var in otherprms.keys():
        setattr(rcomp, var, otherprms[var])
        
    if system.is_driven:
        rcomp.train(tr, *Utr, **methodprms)
    else:
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
    if system.is_driven:
        pre = rcomp.predict(*predargs, **init_cond)
    else:
        pre = rcomp.predict(predargs, **init_cond)
    return pre

def build_params(system, opt_params):
    """ Extract training method parameters and augment reservoir parameters with defaults.
        Parameters
        ----------
        system (templates.System): system the rescomp is being used on
        opt_params: Dictionary of parameters from the optimizer
        
        Returns:
            resprms: Dictionary of reservoir computer parameters
            methodprms: Dictionary of parameters related to the method
            otherprms: Dictionary of all other passed parameters
    """
    METHOD_PRM_NAMES = [
        "window",
        "overlap"
    ]
    RES_PRM_NAMES = {
        "res_sz",
        "activ_f",
        "mean_degree",
        "ridge_alpha",
        "spect_rad",
        "sparse_res",
        "sigma",
        "uniform_weights",
        "gamma",
        "max_weight",
        "min_weight",
        "batchsize",
        "map_initial",
        "delta"
    }
    
    resprms = get_res_defaults()
    methodprms = {}
    otherprms = {}
    for k in opt_params.keys():
        if k in METHOD_PRM_NAMES:
            methodprms[k] = opt_params[k]
        elif k in RES_PRM_NAMES:
            resprms[k] = opt_params[k]
        else:
            otherprms[k] = opt_params[k]
    
    resprms['signal_dim'] = system.signal_dim
    if system.is_driven:
        resprms['drive_dim'] = system.drive_dim
        
    return resprms, methodprms, otherprms
    
#########################
## VPT functions
#########################

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
    """First index i where err[i] > tol. err is assumed to be 1D and tol is a float. If err is never greater than tol, then len(err) is returned."""
    mask = err > tol
    if np.any(mask):
        return np.argmax(mask)
    return len(err)

def get_vptime(system, ts, Uts, pre, vpttol=0.5):
    """
    Valid prediction time for a specific instance.
    """
    err = nrmse(Uts, pre)
    idx = valid_prediction_index(err, vpttol)
    if idx == 0:
        vptime = 0.
    else:
        if system.is_driven:
            vptime = ts[0][idx-1] - ts[0][0]
        else:
            vptime = ts[idx-1] - ts[0]
        
    return vptime
    
#########################
## Setup functions
#########################
    
def get_paramlist(system, train_method, add=None, remove=None):
    """
    Returns a list of sherpa parameters and a list of their names for the given system and training method.
    
    system (templates.System): the system the rescomp is being used on
    train_method (str): 'standard' or 'augmented'
    add (list of sherpa.Parameter): parameters to add
    remove (list of str): names of parameters to remove. Use at own risk.
    """
    parameters = [
        Continuous(name='gamma', range=[0.1, 25]),
        Continuous(name='sigma', range=[0.01, 5.0]),
        Continuous(name='spect_rad', range=[0.1, 25]),
        Continuous(name='ridge_alpha', range=[1e-8, 2], scale='log'),
        Continuous(name='mean_degree', range=[0.1, 5]),
    ]
    augmentedprms = [
        Continuous(name='window', range=[10*system.dt, 0.9*system.train_time]),
        Ordinal(name='overlap', range=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        #Alternative:
        #Continuous(name='overlap', range=[0.0, 0.95])
    ]
    drivenprms = [
        Continuous(name='delta', range=[0.01, 5.0]),
    ]
    if train_method == "augmented":
        parameters += augmentedprms
    if system.is_driven:
        parameters += drivenprms
    if add is not None:
        parameters += add
    if remove is not None:
        parameters = [var for var in parameters if var.name not in remove]
    
    param_names = [var.name for var in parameters]
    
    return parameters, param_names

def loadprior(system_name, parameters, datadir):
    """Load best parameters from random searches (Computed previously).
    Parameters not included are set to a default value found in the parameters files.
    Parameters:
        system_name (str): name of the system type being used
        parameters (list of sherpa.Parameter): list of sherpa parameters in optimizing
        datadir (str): pathname to data directory
    Returns:
        priorprms (List of dictionaries): sets of hyperparameters known to be good"""
    
    #Ensure that all optimization parameters are included, but nothing else
    prior_defaults = {var.name:var.range[0] for var in parameters}
    paramnames = {var.name for var in parameters}
    
    def _clean_prior(prior):
        """Removes unneeded parameters and adds all needed parameters"""
        prior = {**prior_defaults, **prior}
        prior = {key:prior[key] for key in prior if key in paramnames}
        return prior
        
    try:
        with open(datadir + f"{system_name}_prior.pkl", "rb") as file:
            priorprms = pkl.load(file)
            if type(priorprms) is dict:
                #Clean and wrap in a list
                return [_clean_prior(priorprms)]
            elif type(priorprms) is list:
                #Clean each item in the list
                priorprms = [_clean_prior(prms)for prms in priorprms]
                return priorprms
            else:
                print(f"Warning: no correctly-formatted prior data found in {system_name}_prior.pkl", file=sys.stderr)
    except FileNotFoundError as e:
        print(e)
        print("Using empty prior instead.")
    #Return an empty list if we failed to load anything
    return []

#####################
# Miscellaneous
#####################

def bind_function(instance, func, as_name=None):
    """
    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the 
    instance as the first argument, i.e. "self".
    
    Source: https://stackoverflow.com/questions/1015307/python-bind-an-unbound-method#comment8431145_1015405
    """
    if as_name is None:
        as_name = func.__name__
    #Get the function as though it were a method of instance (which it then is)
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method
   











    
