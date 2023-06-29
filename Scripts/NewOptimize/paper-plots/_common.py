"""
Defines some common constants and utilities between the files
"""
import os
import itertools
import numpy as np

# Set some matplotlib properties
from matplotlib import pyplot as plt
plt.rcParams.update({
    'text.usetex': True,
    "font.family": "serif",
    'font.sans-serif': ['Computer Modern Roman'],
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsfonts}',
})

# Wrapper to make plots close in case of exception
def safeclose(func):
    """Closes all plots in the case of an exception"""
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            plt.close('all')
            raise
    return inner

# Helper methods for plots
def clear_3d_axis_labels(ax):
    """
    Cleans up 3d axis plots for when the bounds don't matter
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    ax.set_xticks(np.linspace(*xlim,5), labels=[])
    ax.set_yticks(np.linspace(*ylim,5), labels=[])
    ax.set_zticks(np.linspace(*zlim,5), labels=[])
    #ax.set_xlim(*xlim)
    #ax.set_ylim(*ylim)
    #ax.set_zlim(*zlim)

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
    'icm':TrainMethod('Partial method', 'icm', 'standard', 'activ_f'),
    'windows':TrainMethod('Windowed method', 'windows', 'augmented', 'activ_f'),
}


"""
method_colors = {
    'standard': (0.75, 0.4, 0.3),
    'icm': (0.4, 0.7, 0.5),
    'windows': (0.4, 0.45, 0.9),
}
fill_colors = {
    'standard': 'r',
    'icm': 'g',
    'windows': 'b',
}
fill_alpha = 0.08
"""
method_cmap = plt.get_cmap('gnuplot')
method_colors = {
    'standard': method_cmap(0.13),
    'icm': method_cmap(0.6),
    'windows': method_cmap(0.88),
}
fill_colors = method_colors
fill_alpha = 0.16

PRED_TYPES = {
    'local':PredictionType('Local prediction', 'local', 'continue'),
    'global':PredictionType('Global prediction', 'global', 'random'),
}

ALL_BASE_EXPERIMENTS = lambda: itertools.product(SYSTEMS, TRAIN_METHODS.values(), PRED_TYPES.values())
ALL_BASE_EXPERIMENTS_KEY = lambda: itertools.product(SYSTEMS, TRAIN_METHODS.items(), PRED_TYPES.items())

# Formats for data file names
def hyperparams_file(system, train_method, pred_type, n, c, traintime=None):
    """
    Returns the filename of the hyperparameter file for the given experiment
    Parameters
        system (str)
        train_method (TrainMethod)
        pred_type (PredictionType)
        n (int)
        c (float) - reservoir network mean degree
    """
    if traintime is None:
        path = '../results'
        name = '{}-{}-{}-{}-d{}-n{}.pkl'.format(system, train_method.window_type,
                        pred_type.pred_type, train_method.icm_type, c, n)
    else:
        path = '../traintimes/optimization'
        name = '{}-{}-{}-{}-d{}-n{}-tr{}.pkl'.format(system, train_method.window_type,
                        pred_type.pred_type, train_method.icm_type, c, n, traintime)
    return os.path.join(path, name)
    
def vpts_file(system, train_method, pred_type, n, c, traintime=None):
    """
    Returns the filename of the VPTs file for the given experiment
    Parameters
        system (str)
        train_method (TrainMethod)
        pred_type (PredictionType)
        n (int)
        c (float) - reservoir network mean degree
    """
    if traintime is None:
        path = '../vpt_results'
        name = '{}-{}-{}-{}-d{}-n{}-vpts.pkl'.format(system, train_method.window_type,
                        pred_type.pred_type, train_method.icm_type, c, n)
    else:
        path = '../traintimes/vpts'
        name = '{}-{}-{}-{}-d{}-n{}-tr{}-vpts.pkl'.format(system, train_method.window_type,
                        pred_type.pred_type, train_method.icm_type, c, n, traintime)
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
    