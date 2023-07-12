
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import dill as pickle
import os

from _common import *

def get_data_means():
    """ 
    Loads all of the data. 
    Returns a nested dictionary:
    data[system][train_method][pred_type] gives a list of the means for each network hyperparameter set
    """
    ns = [
        500, 1000, 2000
    ]
    cs = [
        0.1, 1.0, 2.0
    ]
    
    def _single_mean(system, train_method, pred_type, n, c):
        filen = vpts_file(system, train_method, pred_type, n, c)
        with open(filen, 'rb') as file:
            return np.mean(pickle.load(file)[1][pred_type.pred_type])
    
    result = {
        system: {
            train_method.key:{
                pred_type.key: [
                    _single_mean(system, train_method, pred_type, n, c)
                    for n in ns for c in cs
                ]
                for pred_type in PRED_TYPES.values()
            }
            for train_method in TRAIN_METHODS.values()
        }
        for system in SYSTEMS
    }
    
    return result

@safeclose
def create_plots():
    data = get_data_means()
    
    colors = method_colors
    
    for pred_type in PRED_TYPES.values():
        fig, axs = plt.subplots(1,3, figsize=(12,5))
        
        for ax,system in zip(axs, SYSTEMS):
            create_subplot(ax, data, system, pred_type, colors)
            
        # Create the legend
        legend_items = [matplotlib.lines.Line2D([0],[0], color=colors[tr_key], lw = 4, label=train_method.name)
            for (tr_key, train_method) in TRAIN_METHODS.items()]
        axs[-1].legend(handles=legend_items, loc=(0.55, 0.7), fontsize=10.0, framealpha=1.0)
        
    plt.show()
    
    
def create_subplot(ax, data, system, pred_type, colors):
    """
    Creates the subplot and formats it nicely.
    Modifies ax in-place
    Data should be a dictionary of the individual things
    """
    import seaborn
        
    ax.set_title(system.capitalize())
    # Draw all the stuff
    for i, train_method in enumerate(TRAIN_METHODS.values()):
        data_item = data[system][train_method.key][pred_type.key]
        
        # special case to make the kernel size not too small
        if train_method.key == 'standard' and pred_type.pred_type == 'random':
            kernel_scale = {
                'lorenz': 10.0,
                'thomas': 7.0,
                'rossler': 10.0,
            }[system]
        else:
            kernel_scale = 1.0
        
        # Plot the KDE
        seaborn.kdeplot(data=data_item, color=colors[train_method.key], ax=ax, warn_singular=False, fill=False, clip=(0,1000), bw_adjust=kernel_scale, linewidth=2)
        seaborn.kdeplot(data=data_item, color=fill_colors[train_method.key], ax=ax, warn_singular=False, fill=True, clip=(0,1000), bw_adjust=kernel_scale, alpha=fill_alpha, linewidth=0)
        
        ax.set_ylabel(None)
        
    
    
    max_vpt = {
        'lorenz': 6,
        'rossler': 100,
        'thomas': 150,
    }[system]
    
    # Mess around with the bounds
    bounds = ax.axis()
    ax.axis([0, max_vpt, 0, min(8/max_vpt, bounds[3])])
    
    bounds = ax.axis()
    if system == 'lorenz':
        ymax = round(bounds[3], 1)
    elif system == 'rossler':
        ymax = round(bounds[3], 2)
    elif system == 'thomas':
        if pred_type.key == 'local':
            ymax = 0.012
        else:
            ymax = 0.04
    
    ax.set_yticks(ticks=np.linspace(0,ymax,5))
