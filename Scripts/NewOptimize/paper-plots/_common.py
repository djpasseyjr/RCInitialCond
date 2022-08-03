"""
Defines some common constants and utilities between the files
"""
import os
import itertools

# Set some matplotlib properties
from matplotlib import pyplot as plt
plt.rcParams.update({
    'text.usetex': True,
    "font.family": "serif",
    'font.sans-serif': ['Computer Modern Roman'],
    'font.serif': ['Computer Modern Roman'],
})

# Some structs to hold data more cleanly
class TrainMethod:
    def __init__(self, name, key, window_type, icm_type):
        self.name = name
        self.key = key
        self.window_type = window_type
        self.icm_type = icm_type
    def __str__(self):
        return self.name
class PredictionType:
    def __init__(self, name, key, pred_type):
        self.name = name
        self.key = key
        self.pred_type = pred_type
    def __str__(self):
        return self.name

SYSTEMS = [
    'lorenz',
    'rossler',
    'thomas',
]

# Train methods
TRAIN_METHODS = {
    'standard':TrainMethod('Standard method', 'standard', 'standard', 'random'),
    'icm':TrainMethod('ICM method', 'icm', 'standard', 'activ_f'),
    'windows':TrainMethod('Windowed method', 'windows', 'augmented', 'activ_f'),
}

PRED_TYPES = {
    'local':PredictionType('Local prediction', 'local', 'continue'),
    'global':PredictionType('Global prediction', 'global', 'random'),
}

ALL_BASE_EXPERIMENTS = lambda: itertools.product(SYSTEMS, TRAIN_METHODS.values(), PRED_TYPES.values())
ALL_BASE_EXPERIMENTS_KEY = lambda: itertools.product(SYSTEMS, TRAIN_METHODS.items(), PRED_TYPES.items())

# Formats for data file names
def hyperparams_file(system, train_method, pred_type, n, c):
    """
    Returns the filename of the hyperparameter file for the given experiment
    Parameters
        system (str)
        train_method (TrainMethod)
        pred_type (PredictionType)
        n (int)
        c (float) - reservoir network mean degree
    """
    path = '../results'
    name = '{}-{}-{}-{}-d{}-n{}.pkl'.format(system, train_method.window_type,
                    pred_type.pred_type, train_method.icm_type, n, c)
    return os.path.join(path, name)
    
def vpts_file(system, train_method, pred_type, n, c):
    """
    Returns the filename of the VPTs file for the given experiment
    Parameters
        system (str)
        train_method (TrainMethod)
        pred_type (PredictionType)
        n (int)
        c (float) - reservoir network mean degree
    """
    path = '../vpt_results'
    name = '{}-{}-{}-{}-d{}-n{}-vpts.pkl'.format(system, train_method.window_type,
                    pred_type.pred_type, train_method.icm_type, c, n)
    return os.path.join(path, name)
    
def attractorsamples_file(system, train_method, pred_type, n, c):
    """
    Returns the filename of the attractor samples file for the given experiment
    Parameters
        system (str)
        train_method (TrainMethod)
        pred_type (PredictionType)
        n (int)
        c (float) - reservoir network mean degree
    """
    path = '../attractor_results'
    name = '{}-{}-{}-{}-d{}-n{}-attractor.pkl'.format(system, train_method.window_type,
                    pred_type.pred_type, train_method.icm_type, c, n)
    return os.path.join(path, name)