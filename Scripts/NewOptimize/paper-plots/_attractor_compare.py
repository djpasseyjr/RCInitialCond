from matplotlib import pyplot as plt, colors, cm
import numpy as np
import dill as pickle

from _common import *

def _single_plot(data, system):
    """Accepts the data dictionary as constructed in create_plots()"""
    # Per-system parameters
    max_vpts = {
        'lorenz':   10.,
        'rossler':  100.,
        'thomas':   100.,
    }
    ticks = {
        'lorenz':   [0, 0.1, 1, 10],
        'rossler':  [0, 1, 10, 100],
        'thomas':   [0, 1, 10, 100],
    }
    thresh = {
        'lorenz':   1e-2,
        'rossler':  1e-1,
        'thomas':   1e-1,
    }
    
    colormap = 'plasma'
    norm = colors.SymLogNorm(vmin=0, vmax=max_vpts[system], linthresh=thresh[system], linscale=thresh[system]*1e-2, base=10)
    
    # Create the figure/axes
    fig = plt.figure(figsize=(12,4))
    axs = []
    for i,train_method in enumerate(TRAIN_METHODS.values()):
        ax = fig.add_subplot(1,3,1+i, projection='3d')
        axs.append(ax)
        # Extract data
        samples = data[system][train_method.key]
        x,y,z = samples[:,0], samples[:,1], samples[:,2]
        vpts = samples[:,-1]
        # Colors
        c = plt.get_cmap(colormap)(norm(vpts))
        
        ax.scatter(x,y,z,c=c, s=2., marker='.')
        
        clear_3d_axis_labels(ax)
        
        ax.set_title(train_method.name)
        
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=axs, shrink=0.7, label="Valid Prediction Time", ticks=ticks[system])
    
    
    
    return fig

def create_plots(n=1000, c=1.0):
    # Which attractor plot to use
    which = {
        'lorenz': {
            'standard': 0,
            'icm':      0,
            'windows':  0,
        },
        'rossler': {
            'standard': 0,
            'icm':      0,
            'windows':  0,
        },
        'thomas': {
            'standard': 0,
            'icm':      0,
            'windows':  0,
        },
    }
    
    # First, load the data
    def _get_data(system, train_method):
        filename = attractorsamples_file(system, train_method, PRED_TYPES['global'], n, c)
        with open(filename, 'rb') as file:
            return pickle.load(file)['results'][which[system][train_method.key]]['samples']
    
    data = {
        system: {
            tr_key: _get_data(system, train_method) for tr_key, train_method in TRAIN_METHODS.items()
        }
        for system in SYSTEMS
    }
    
    try:
        # Now, do the plots
        for system in SYSTEMS:
            _single_plot(data, system)
        plt.show()
    except Exception as e:
        plt.close('all')
        raise

