import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import dill as pickle
import os
from glob import glob

from _common import *


def logspace(x1, x2, n_points):
    return np.exp(np.linspace(np.log(x1), np.log(x2), n_points))

def get_system_data(system):
    windows = {
        'lorenz':logspace(0.02, 6.6, 30),
        'rossler':logspace(0.1, 165, 30),
        'thomas':logspace(1, 660, 30),
    }[system]
    overlaps = np.linspace(0,0.95,25)
    windows_dict = {val:idx for idx,val in enumerate(windows)}
    overlaps_dict = {val:idx for idx,val in enumerate(overlaps)}
    
    mean_vpts = np.full((len(windows), len(overlaps)), np.nan)
    
    for filename in glob(f"../results_grid_search/{system}*.pkl"):
        if filename.endswith("config.pkl"):
            continue
        with open(filename, 'rb') as file:
            (
                (system, window, overlap, aug_type, pred_type, init_cond, mean_degree, n),
                best_params, 
                vpt_results
            ) = pickle.load(file)
            
            mean_vpts[
                windows_dict[window], overlaps_dict[overlap]
            ] = np.mean(vpt_results['random'])
    
    
    return *np.meshgrid(windows, overlaps), mean_vpts.T

def create_system_plot(system, ax):
    windows, overlaps, mean_vpts = get_system_data(system)
    
    cmap = 'plasma'
    norm = matplotlib.colors.Normalize(vmin=0, vmax=np.max(mean_vpts[np.isfinite(mean_vpts)]))
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    ax.set_xscale('log')
    
    ax.pcolormesh(windows, overlaps, mean_vpts, shading='nearest', norm=norm, cmap=cmap)
    #ax.tricontourf(windows, overlaps, mean_vpts)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 0.95])
    
    ax.set_title(system.capitalize())
    
    ax.set_xlabel('Window size')
    
    plt.colorbar(mappable=sm, ax=ax)
    
@safeclose
def create_plots():
    fig, axs = plt.subplots(1,3, figsize=(10,3.2))
    
    for system, ax in zip(['lorenz', 'rossler', 'thomas'], axs):
        create_system_plot(system, ax)
    axs[0].set_ylabel('Overlap')
    plt.tight_layout()
    plt.show()