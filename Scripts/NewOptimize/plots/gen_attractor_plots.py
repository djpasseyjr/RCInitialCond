import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm
import dill as pickle
import os
import sys
from tqdm import tqdm

def get_name(filepath):
    return filepath.replace('\\','/').split('/')[-1][:-4]

def get_system(name):
    return name.split('-')[0]

def main(overwrite=True, verbose=False, extension='pdf', **plot_params):
    """
    Arguments:
        overwrite: whether to re-create existing plots
        verbose: whether to print output as it goes along
        extension: file format to save plots in
    All other arguments are passed to ax.scatter.
    """
    result_dir = "attractor"
    os.makedirs(result_dir, exist_ok=True)
    
    data_dir = '../attractor_results'
    
    max_vpts = {
        'lorenz':0.0,
        'thomas':0.0,
        'rossler':0.0
    }
    # get the maximum vpts for each attractor
    # we could load all of the data at once, which might be faster,
    #   but there's enough that I'd be apprehensive.
    if verbose:
        print("Getting max vpts...")
    for filename in glob(os.path.join(data_dir, "*.pkl")):
        # get rid of intermediate paths and the extension
        name = get_name(filename)
        system = get_system(name)
        
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        
        # Get the best vpt from each system
        best = [max_vpts[system]] + [np.max(item['samples'][:,-1]) for item in data['results']]
        max_vpts[system] = np.max(best)
    
    # make the plots.
    filenames = glob(os.path.join(data_dir, "*.pkl"))
    if verbose:
        print("Generating plots...")
        filenames = tqdm(filenames)
        progress_bar = filenames
    for filename in filenames:
        # get rid of intermediate paths and the extension
        name = get_name(filename)
        system = get_system(name)
        if verbose:
            progress_bar.set_description(name[:-10])
        
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        
        #Make the plots
        for i,item in enumerate(data['results']):
            dest_name = os.path.join(result_dir, f'{name}-{i}.{extension}')
            if not overwrite:
                # check if the file exists
                if os.path.exists(dest_name):
                    continue
            make_attractor_plot(dest_name, data['experiment'], item['samples'], 
                        max_vpts[system], **plot_params)
    
def make_attractor_plot(dest_name, experiment, samples, max_vpt, cmap='plasma', figsize=(5,5), **plot_params):
    
    x,y,z = samples[:,0], samples[:,1], samples[:,2]
    vpts = samples[:,-1]
    
    # get the colors
    norm = colors.SymLogNorm(vmin=0, vmax=max_vpt, linthresh=1e-3, linscale=1e-1, base=10)
    #norm = colors.Normalize(vmin=0, vmax=max_vpt)
    c = plt.get_cmap(cmap)(norm(vpts))
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1,projection='3d')
    
    ax.scatter(x,y,z, c=c, **plot_params)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
    
    plt.tight_layout()
    plt.savefig(dest_name)
    plt.close(fig)
    
if __name__=="__main__":
    overwrite = True
    if len(sys.argv) == 2:
        overwrite = bool(sys.argv[1])
    elif len(sys.argv) > 2:
        raise ValueError("Too many arguments")
    #pdf will give you higher quality images but takes about 5x as long
    main(verbose=True, overwrite=overwrite, figsize=(5,4), s=2., marker='.', extension='png')