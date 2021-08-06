#!/bin/python3

if __name__!="__main__":
    #maybe should print an error message
    exit()
    
#######################
# Parse arguments. Information for that is stored in script_args.py.
#######################
from script_args import get_parser
parser = get_parser()
#If invalid/insufficient arguments are given, execution stops here
args = parser.parse_args()


import optimizer_controller as opt
import optimizer_systems
from optimizer_functions import loadprior

import dill as pkl

from os import mkdir
from datetime import datetime


if __name__=="__main__":
    #######################
    ## Process arguments
    #######################
    
    if args.debug:
        print("Running in debug mode")
    
    if not args.no_timestamp:
        TIMESTAMP = ''
    else:
        TIMESTAMP = "{:%y%m%d%H%M%S}".format(datetime.now())
    
    #Load the system
    system = optimizer_systems.get_system(args.system)
    
    #Set up the data directory
    resultsdir = args.resultsdir
    if resultsdir is None:
        resultsdir = "_".join((system.name, args.map_initial, args.prediction_type, args.method, TIMESTAMP))
        if args.debug:
            resultsdir = "TEST-" + resultsdir
        #resultsdir = DATADIR + system.name + "/" + resultsdir
        resultsdir = optimizer_systems.DATADIR + "results/" + resultsdir
    #Make sure the data directory exists
    try:
        mkdir(resultsdir) #questionable
    except FileExistsError:
        pass
    except FileNotFoundError:
        raise FileNotFoundError("Data directory must already exist, except possibly the lowest subdirectory")

    is_parallel = (args.parallel or (args.profile is not None))
    
    #Load alternate reservoir ODE and parameters
    if args.ode is not None:
        res_ode = optimizer_systems.load_from_file(args.ode, 'res_odes')
        add_opt_params = res_ode.pop('opt_parameters', None)
        add_const_params = res_ode.pop('parameters', dict())
        rm_params = res_ode.pop('remove', None)
        
    else:
        res_ode = None
        add_opt_params = None
        rm_params = None
        add_const_params = dict()

    #######################
    ## Optimize and test
    #######################
    
    #Initialize the thing that runs the optimization
    rcoptimizer = opt.ResCompOptimizer(args.system, args.map_initial, args.prediction_type, args.method,
                results_directory=resultsdir, res_ode=res_ode,
                parallel=is_parallel, parallel_profile=args.profile,
                add_params=add_opt_params, rm_params=rm_params, **add_const_params)
    if args.debug:
        print("Initialization successful")

    if not args.debug:
        opt_ntrials = args.opt_ntrials
        opt_vpt_reps = args.opt_vpt_reps
        test_ntrials = args.test_ntrials
        lyap_reps = args.test_lyap_reps
        saved_orbits = args.saved_orbits
    else: 
        opt_ntrials = 3
        opt_vpt_reps = 2
        test_ntrials = 1
        lyap_reps = 2
        saved_orbits = 0
    
    #Run the optimization
    rcoptimizer.run_optimization(opt_ntrials, opt_vpt_reps, sherpa_dashboard=args.use_dashboard)
    print("Optimization ran succesfully")
    #Run the tests
    results = rcoptimizer.run_tests(test_ntrials, lyap_reps)
    if saved_orbits > 0:
        orbits = rcoptimizer.generate_orbits(saved_orbits)
        results['orbits'] = orbits
    
    #Add some additional information to the results
    results["experiment"] = (args.system, args.map_initial, args.prediction_type, args.method)
    results["opt_parameters"] = rcoptimizer.get_best_result()
    results["is_debug"] = args.debug
    results['timestamp']= TIMESTAMP
    
    #######################
    ## Save results
    #######################
    
    results_filename = "-".join((args.system, args.map_initial, args.prediction_type, args.method, TIMESTAMP)) + ".pkl"
    if args.debug:
        results_filename = "TEST-" + results_filename
    with open(resultsdir + "/" + results_filename, 'wb') as file:
        pkl.dump(results, file)

    print("Testing ran successfully")
    print(f"Results written to {resultsdir}/{results_filename}.")
    