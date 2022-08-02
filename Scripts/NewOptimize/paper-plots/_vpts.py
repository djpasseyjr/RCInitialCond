
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import dill as pickle
import seaborn
import os

from _common import *

def fetch_data(n, c):
    """
    Returns the VPT data corresponding to the given n and c
    """
    results = {
        'local':{system:dict() for system in SYSTEMS},
        'global':{system:dict() for system in SYSTEMS},
    }
    for system, (tr_key, train_method), (pr_key, pred_type) in ALL_BASE_EXPERIMENTS_KEY():
        filename = vpts_file(system, train_method, pred_type, n, c)
        with open(filename, 'rb') as file:
            # Only load the vpts
            data = pickle.load(file)[1]
            results[pr_key][system][tr_key] = data
    
    return results
    
def create_vpt_plots(n=1000, c=1.0, figsize=(12,5)):
    # Fetch the data
    data = fetch_data(n, c)
    
    colors = dict(zip(
        TRAIN_METHODS.keys(),
        ['r', 'g', 'b']
    ))
    
    try:
        for pr_key, pred_type in PRED_TYPES.items():
            # Create the figure
            fig, axs = plt.subplots(1,3, figsize=figsize)
            plt.suptitle('{} accuracy (VPT)'.format(pred_type.name), fontsize=16.0)
            
            # Draw each subplot
            for i, system in enumerate(SYSTEMS):
                create_subplot(axs[i], data[pr_key][system], system, pred_type, colors)
                axs[i].set_title(str.capitalize(system), fontsize=14.0)
            
            # Create the legend
            legend_items = [matplotlib.lines.Line2D([0],[0], color=colors[tr_key], lw = 4, label=train_method.name)
                for (tr_key, train_method) in TRAIN_METHODS.items()]
            axs[-1].legend(handles=legend_items, loc=(0.55, 0.7), fontsize=12.0, framealpha=1.0)
                
            #Adjust positioning
            plt.subplots_adjust(left=0.06, right=0.92)
        plt.show()
    except Exception as e:
        # in case of an error clean up created plots
        plt.close('all')
        raise

def create_subplot(ax, data, system, pred_type, colors):
    """
    Creates the subplot and formats it nicely.
    Modifies ax in-place
    Data should be a dictionary of the individual things
    """
    # Draw all the stuff
    max_vpt = 0
    for i, (tr_key, train_method) in enumerate(TRAIN_METHODS.items()):
        data_item = data[tr_key][pred_type.pred_type]
        
        # special case to make the kernel size not too small
        if tr_key == 'standard' and pred_type.pred_type == 'random':
            kernel_scale = {
                'lorenz': 10.0,
                'thomas': 7.0,
                'rossler': 10.0,
            }[system]
        else:
            kernel_scale = 1.0
        
        # Plot the KDE
        seaborn.kdeplot(data=data_item, color=colors[tr_key], ax=ax, warn_singular=False, fill=True, clip=(0,1000), bw_adjust=kernel_scale, alpha=0.1, linewidth=2)
        
        ax.set_ylabel(None)
        
        # Plot the mean
        # Manually set up so they line up with the KDE plot
        # Changing pretty much anything would necessitate changing
        #   most of these
        mean_plot_height = {
            ('lorenz', 'standard', 'local'):    0.60,
            ('lorenz', 'standard', 'global'):   0.999,
            ('lorenz', 'icm', 'local'):         0.55,
            ('lorenz', 'icm', 'global'):        0.999,
            ('lorenz', 'windows', 'local'):     0.61,
            ('lorenz', 'windows', 'global'):    0.35,
            ('rossler', 'standard', 'local'):   0.77,
            ('rossler', 'standard', 'global'):  0.999,
            ('rossler', 'icm', 'local'):        0.77,
            ('rossler', 'icm', 'global'):       0.999,
            ('rossler', 'windows', 'local'):    0.72,
            ('rossler', 'windows', 'global'):   0.37,
            ('thomas', 'standard', 'local'):    0.82,
            ('thomas', 'standard', 'global'):   0.999,
            ('thomas', 'icm', 'local'):         0.79,
            ('thomas', 'icm', 'global'):        0.64,
            ('thomas', 'windows', 'local'):     0.72,
            ('thomas', 'windows', 'global'):    0.43,
        }[(system, tr_key, pred_type.key)]
        
        ax.axvline(x=np.mean(data_item), color=colors[tr_key], linestyle="--",alpha=0.8, ymax=mean_plot_height)
        
        max_vpt = max(max_vpt, np.max(data_item))
    
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
    