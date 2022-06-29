
import numpy as np
from parameters.ott_params import *
import opt_then_test

opt_then_test.SYSTEM = "rossler"
opt_then_test.MAP_INITIAL = "relax"
opt_then_test.PREDICTION_TYPE = "random"
opt_then_test.METHOD = "augmented"

parameters = {'gamma': 25.0, 'sigma': 5.0, 'spect_rad': 25.0, 'ridge_alpha': 2.0, 'mean_degree': 5.0, 'window': 0.1, 'overlap': 0.0, 'res_sz': 1000, 'activ_f': lambda x: 1/(1+exp(-1*x)), 'sparse_res': True, 'uniform_weights':True, 'signal_dim': 3, 'max_weight': 2, 'min_weight': 0, 'batchsize': 2000, 'map_initial': 'relax'}

print("Running...")
opt_then_test.vpt('rossler','random','augmented', **parameters)