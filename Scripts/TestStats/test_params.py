
import numpy as np
import rescomp as rc
from rescomp import optimizer as rcopt
import dill as pkl
import sys

#def get_parameters(sysname, icmap, predtype, augtype):
def get_parameters(sysname, icmap, augtype):
    #TODO get different/more complete parameter set; also include continue/random?
    params = {
        ('lorenz', 'activ_f', 'augmented'): {'gamma': 9.887252061575511,
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
          'mean_degree': 0.991321866748073}
    }
    return params[(sysname, icmap, augtype)]

#def main(system, icmap, predtype, augtype, n_trials, lyap_reps=20, parallel_profile=None, result_dir=""):
def main(system, icmap, augtype, n_trials, lyap_reps=20, parallel_profile=None, result_dir=""):
    opt = rcopt.ResCompOptimizer(system, icmap, 'continue', augtype, parallel_profile=parallel_profile)
    
    # Run the thing
    results = opt.run_tests(n_trials, lyap_reps, parameters=get_parameters(system, icmap, augtype))
    
    # Save results
    result_filename = result_dir + "{}-{}-{}.pkl".format(system,icmap,augtype)
    with open(result_filename, 'wb') as file:
        pkl.dump(results, file)

if __name__ == "__main__":
    n_args = 6
    system, icmap, augtype, n_trials, result_dir = sys.argv[1:n_args]
    n_trials = int(n_trials)
    if len(sys.argv) > n_args:
        parallel_profile=sys.argv[n_args]
    else:
        parallel_profile=None
    main(system, icmap, augtype, n_trials, parallel_profile=parallel_profile, result_dir=result_dir)
    