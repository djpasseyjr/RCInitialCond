#optimizer_controller.py
"""Class to perform hyperparameter optimization on a reservoir computer"""

from ipyparallel import Client
import sherpa
import numpy as np

import optimizer_systems
import templates
import optimizer_functions as functions

class ResCompOptimizer:
    """
    Attributes:
        
        parallel (bool): whether using parallelization
        results_directory
        prior
        opt_parameters (list): sherpa parameters to optimize over
        study
        prediction_type
        
        If using parallelization:
            dview (ipp.dview): view into ipyparallel nodes
            node_count (int): number of nodes in use
    """
    def __init__(self, system, map_initial, prediction_type, method, res_ode=None,
                add_params=None, rm_params=None, results_directory="",
                parallel=False, parallel_profile=None, **res_params):
        """
        Arguments:
            system (string or template.System): the system to use. If not a template.System object, will load the corresponding one from defaults.
            map_initial (string): initial condition mapping for reservoir computer to use
            prediction_type (string): 'random' or 'continue'; prediction type to use while optimizing.
            method (string): training method; 'standard' or 'augmented'
            
        Optional arguments:
            results_directory (str): pathname of where to store optimization results
            res_ode (dict->callable): dictionary containing functions 'res_ode' and 'trained_res_ode' to replace the reservoir computer's usual internal ODE
            
            add_params: list of sherpa.Parameter objects to include in optimization
            rm_params (list of str): names of optimization parameters to remove
            
            parallel (bool): whether to use parallelization. Default false
            parallel_profile (str or None): when using parallelization, the ipyparallel profile to connect to. None probably leads to default behavior, although I haven't actually tested this.
            
            All other keyword arguments are passed to the reservoir computers created.
        
        When specifying an alternate reservoir ODE, the dictionary should map 'res_ode' and 'trained_res_ode' to functions that have the same signatures as the corresponding function in ResComp or DrivenResComp.
        It is also possible to add or remove parameters from optimization if they are (or are not) used in the new reservoir ODE. To add parameters, place a list of sherpa.Parameter objects in a list under key 'parameter'. To remove parameters, place a list of the parameters' names in a list under key 'remove'.
            
        """
        if not isinstance(system, templates.System):
            self.system = optimizer_systems.get_system(system)
        else:
            self.system = system
        self.prediction_type = prediction_type
        
        self.parallel = parallel
        self.results_directory = results_directory
        
        self.opt_parameters, self.opt_param_names = functions.get_paramlist(self.system, method,
                    add=add_params, remove=rm_params)
        self.prior = functions.loadprior(self.system.name, self.opt_parameters, optimizer_systems.DATADIR)
        self.study = None
        
        self.res_params = {**res_params, 'res_ode':res_ode, 'map_initial':map_initial}
        
        if parallel:
            self._initialize_parallelization(parallel_profile)
            
    def run_optimization(self, opt_ntrials, vpt_reps, sherpa_dashboard=False):
        """Runs the optimization process.
        
        Arguments:
            opt_ntrials (int): number of hyperparameter configurations to attempt
            vpt_reps (int): number of times to try each parameter set
            sherpa_dashboard (bool): whether to use the sherpa dashboard. Default false."""
        self._initialize_sherpa(opt_ntrials, sherpa_dashboard)
        for trial in self.study:
            try:
                exp_vpt, stdev = self.run_single_vpt_test(vpt_reps, trial.parameters)
            except Exception as e:
                #print relevant information for debugging
                print("Trial parameters at error:", trial.parameters)
                print("Other parameters:", self.res_params)
                raise e
            self.study.add_observation(trial=trial,
                              objective=exp_vpt,
                              context={'vpt_stdev':stdev})
            self.study.finalize(trial)
            self.study.save(self.results_directory)
    
    def run_tests(self, test_ntrials, lyap_reps=20, parameters=None):
        """
        Runs tests using the given parameters.
        If not passed, uses the optimized hyperparameters.
        
        Tests the reservoir computer for the following quantities:
            -continue prediction vpt
            -random prediction vpt
            -lyapunov exponent of reservoir's recreation of the system
            For systems governed by a differential equation:
            -derivative fit of continued prediction
            -derivative fit of random prediction
        All of these results are returned in a dictionary mapping the names to the attributes.
        
        Arguments:
            test_ntrials (int): number of times to test the parameter set
            parameters (dict): the hyperparameter set to use. If not specified, uses the best result from optimizing.
        """
        if parameters is None:
            parameters = self.get_best_result()
            
        if self.parallel:
            results = self._run_n_times_parallel(test_ntrials, functions.test_all,
                        self.system, lyap_reps=lyap_reps, **self.res_params, **parameters)
            #Collapse into single list of outputs
            results = [item for sublist in results for item in sublist]
        else:
            results = self._run_n_times(test_ntrials, functions.test_all,
                        self.system, lyap_reps=lyap_reps, **self.res_params, **parameters)
        
        #Collect results into a dictionary
        results_dict = {name:[] for name in {'continue','random','lyapunov','cont_deriv_fit','rand_deriv_fit'}}
        for item in results:
            cont_vpt, rand_vpt, lyap, cont_df, rand_df = item
            results_dict["continue"].append(cont_vpt)
            results_dict["random"].append(rand_vpt)
            results_dict["lyapunov"].append(lyap)
            if self.system.is_diffeq:
                results_dict["cont_deriv_fit"].append(cont_df)
                results_dict["rand_deriv_fit"].append(rand_df)
        return results_dict
    
    def generate_orbits(self, n_orbits, parameters=None):
        """
        Trains a reservoir computer and has it predict, using the given hyperparameters
        If parameters are not specified, uses the optimized hyperparameters.
        
        Arguments:
            n_orbits (int): number of times to test the parameter set
            parameters (dict): the hyperparameter set to use. If not specified, uses the best result from optimizing.
        
        Returns:
            a list of orbit data, where each entry is a tuple consisting of (tr, Utr, ts, Uts, pre).
        """
        if parameters is None:
            parameters = self.get_best_result()
            
        if self.parallel:
            results = self._run_n_times_parallel(n_orbits, functions.create_orbit,
                        self.system, self.prediction_type, **self.res_params, **parameters)
            #Collapse into single list of outputs
            results = [item for sublist in results for item in sublist]
        else:
            results = self._run_n_times(n_orbits, functions.create_orbit,
                        self.system, self.prediction_type, **self.res_params, **parameters)
        
        return results
    
    def run_single_vpt_test(self, vpt_reps, trial_params):
        """Returns the mean and standard deviation of valid prediction time (VPT) resulting from the current and specified parameters"""
        if self.parallel:
            vpts = self._run_n_times_parallel(vpt_reps, functions.vpt,
                        self.system, self.prediction_type, **self.res_params, **trial_params)
        else:
            vpts = self._run_n_times(vpt_reps, functions.vpt,
                        self.system, self.prediction_type, **self.res_params, **trial_params)
        
        return np.mean(vpts), np.std(vpts)
        
    def get_best_result(self):
        """Returns the best parameter set found in the previous optimization attempt."""
        if self.study is None:
            raise RuntimeError("must run optimization before getting results!")
        result = self.study.get_best_result()
        #Clean non-parameters from the dictionary
        return {key:result[key] for key in result.keys() if key in self.opt_param_names}
    
    #### Internal functions ####
        
    def _run_n_times(self, n, func, *args, **kwargs):
        """
        Calls func(*args, **kwargs) n times, and returns the result as a list.
        """
        return [func(*args, **kwargs) for _ in range(n)]
        
    def _run_n_times_parallel(self, n, func, *args, **kwargs):
        """
        Calls func(*args, **kwargs) (at least) n times total between the ipyparallel nodes, and returns the result as a list.
        The function is called the same number of times on each node, and guaranteed to be called at least a total number of n times, although generally will be slightly more.
        """
        run_ct = int(np.ceil(n / self.node_count))
        result = self.dview.apply(lambda k, *args, **kwargs: [func(*args, **kwargs) for _ in range(k)],
                        run_ct, *args, **kwargs)
        return result
        
    def _initialize_parallelization(self, parallel_profile):
        """Helper function to set up parallelization.
        Connects to the ipyparallel client and initializes an engine on each node."""
        if not self.parallel:
            #this shouldn't happen in general
            raise RuntimeError("parallelization cannot be set up if not enabled")
        client = Client(profile=parallel_profile)
        self.dview = client[:]
        self.dview.use_dill()
        self.dview.block = True #possibly can remove, but would require modifying other parts
        self.node_count = len(client.ids)
        print(f"Using multithreading; running on {self.node_count} engines.")
    
    def _initialize_sherpa(self, opt_ntrials, sherpa_dashboard=False):
        """Initializes the sherpa study used internally"""
        algorithm = sherpa.algorithms.GPyOpt(max_num_trials=opt_ntrials, initial_data_points=self.prior)
        self.study = sherpa.Study(parameters=self.opt_parameters,
                         algorithm=algorithm,
                         disable_dashboard=(not sherpa_dashboard),
                         lower_is_better=False)
                         
