import numpy as np
from glob import glob
from matplotlib import pyplot as plt
import dill as pickle
import os
from tqdm import tqdm

def get_name(filepath):
    name = filepath.replace('\\','/').split('/')[-1][:-4]
    if name.endswith('-vpts'):
        name = name[:-5]
    return name

def get_system(name):
    return name.split('-')[0]

def gen_plot(ax, data_dict, is_rand, max_vpt, colors, n_bins=20, label_name=True, label_mean=True, **plot_params):
    data = {key:data_dict[key][is_rand][is_rand] for key in data_dict if is_rand in data_dict[key].keys()}
    bins = np.linspace(0, max_vpt, n_bins)
    
    for key in data:
        mean = np.mean(data[key])
        
        # Generate the label
        label = ''
        if label_name:
            label = '-'.join(key)
        if label_mean:
            if len(label) > 0:
                label += '; '
            label += r"$\mu = {:.2f}$".format(mean)
        
        ax.hist(data[key], bins=bins, color=colors[key], edgecolor=colors[key], label=label, **plot_params)
        ax.axvline(x=mean, color=colors[key],linestyle="--",alpha=0.8)
    

def main(verbose=False, extension='pdf', **plot_params):
    result_dir = "histograms"
    os.makedirs(result_dir, exist_ok=True)
    
    colors = {
        ('augmented','activ_f'):'b',
        ('standard','activ_f'):'g',
        ('standard','random'):'r',
    }
    
    data_dir = '../vpt_results'
    data = dict()
    
    if verbose:
        print("Loading data...")
    # Load the data
    for filename in glob(os.path.join(data_dir, '*.pkl')):
        name = get_name(filename)
        parts = name.split('-')
        
        outer_key = (parts[0], *parts[4:])
        inner_key = (parts[1], parts[3])
        is_rand = parts[2]
        
        if not outer_key in data.keys():
            data[outer_key] = dict()
            
        if not inner_key in data[outer_key].keys():
            data[outer_key][inner_key] = dict()
            
        with open(filename, 'rb') as file:
            # first item is just the experiment/parameters
            data[outer_key][inner_key][is_rand] = pickle.load(file)[1]
        
    # Generate plots
    keys = data.keys()
    if verbose:
        print("Generating plots...")
        keys = tqdm(keys)
        progress_bar = keys
    
    for key in keys:
        if verbose:
            progress_bar.set_description('-'.join(key))
            
        dest_file = os.path.join(result_dir, '{}-{}-{}.{}'.format(*key, extension))
    
        fig, ax = plt.subplots(2,1, figsize=(6,8))
        # determine the largest vpt for plotting reasons
                    
        max_vpt = max([np.max(data[key][subkey][is_rand][is_rand]) for subkey in data[key].keys() for is_rand in {'continue','random'} if is_rand in data[key][subkey].keys()])
        
        ax[0].set_title('Continue predictions')
        gen_plot(ax[0], data[key], 'continue', max_vpt, colors, **plot_params)
        ax[0].legend(loc='upper right')
        ax[1].set_title('Random predictions')
        gen_plot(ax[1], data[key], 'random', max_vpt, colors, label_name=False, **plot_params)
        ax[1].legend(loc='upper right')
        fig.savefig(dest_file)
        plt.close(fig)
        
    
if __name__=="__main__":
    main(extension='png', verbose=True, alpha=0.3, n_bins=40)
    