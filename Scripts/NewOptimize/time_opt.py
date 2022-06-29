
import rescomp
from rescomp import optimizer as rcopt
import time
import ipyparallel
import os
from contextlib import contextmanager

def _time_func(f, *args, **kwargs):
    """
    Returns a tuple with the time of execution (in s) and the output of the function.
    """
    start = time.perf_counter()
    val = f(*args, **kwargs)
    end = time.perf_counter()
    return end-start, val

@contextmanager
def open_secure(filename, *args, interval=0.2, **kwargs):
    """
    Opens a file in a way that can be used safely by tasks running in parallel
    """
    # Make sure the file isn't being used elsewhere
    lockname = filename + ".lock"
    while True:
        while os.path.exists(lockname):
            time.sleep(interval)
        try:
            # Attempt to lock it
            with open(lockname,'x'):
                pass
        except FileExistsError:
            continue
        break
    try:
        with open(filename, *args, **kwargs) as file:
            yield file
    finally:
        # Finish
        os.remove(lockname)
    
    

def accumulate_tikhanov(rc, t, U, window=None, overlap=0):
    idxs = rc._partition(t, window, overlap=overlap)
    for start, end in idxs:
        ti = t[start:end]
        Ui = U[start:end, :]
        rc.update_tikhanov_factors(ti, Ui)

def time_training(system, **rescomp_params):
    # Parts to time:
    # - Reservoir response, accumulate Rhat, Yhat
    # - Calculate W_out
    # - Generate test signal response
    # Get train/test data
    
    tr, Utr, ts, Uts = system.get_train_test_data(True)
    # Extract parameters
    resprms, methodprms, otherprms = rcopt.build_params(system, rescomp_params)
    
    start = time.perf_counter()
    # Initialize
    rc, init_time = _time_func(rescomp.ResComp, **resprms)
    # Train; the next 3 lines do the equivalent of rc.train(...)
    _, train_response_time = _time_func(accumulate_tikhanov, rc, tr, Utr)
    # W_out
    rc.W_out, wout_time = _time_func(rc.solve_wout)
    rc.is_trained = True
    
    # Test response
    pred, pred_time = _time_func(rc.predict, tr)
    
    end = time.perf_counter()
    tot_time = end-start
    
    return tot_time, init_time, train_response_time, wout_time, pred_time
    
def main():
    # As far as what to do:
    #   -test optimized hyperparameter times
    #   -test varying one parameter to make it clearer to what extent
    #       they affect runtime
    # General approach:
    #   -keep a list (set?) of jobs to do
    #   -use ipyparallel asynchronously
    #   -save progress periodically
    pass
    
    