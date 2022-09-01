from matplotlib import pyplot as plt, colors, cm
import numpy as np
import dill as pickle

from _common import *

def attractor_with_train_signal(system, train_method, c, n, idx):
    import rescomp as rc
    if isinstance(train_method, str):
        train_method = TRAIN_METHODS[train_method]
    # Load the data
    pred_type = 'global'
    data_filen = attractorsamples_file(system, train_method, PRED_TYPES[pred_type], n, c)
    with open(data_filen, 'rb') as file:
        data = pickle.load(file)['results'][idx]
        
    if train_method.key == 'windows':
        # Get window parameters
        param_filen = hyperparams_file(system, train_method, PRED_TYPES[pred_type], n ,c)
        with open(param_filen, 'rb') as file:
            params = pickle.load(file)[1]
    else:
        # only need to load these for windowed
        params = None
    
    samples = data['samples']
    u0 = data['train_initial']
    
    train_times = {
        'lorenz': 6.6,
        'rossler': 165.,
        'thomas': 660.,
    }
    
    # Colorbar parameters
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
    
    
    # Get the training signal
    tr, Utr = rc.orbit(system, duration=train_times[system], initial=u0)

    # Plotting time
    try:
        fig = plt.figure(figsize=(5,8))
        ax_tr = fig.add_axes([0.0, 0.4,1.0,0.6], projection='3d')
        ax_tr.patch.set_alpha(0.0)
        ax_tr.axis('off')
        ax_samples = fig.add_axes([0.0, 0.0,1.0,0.6], projection='3d')
        ax_samples.patch.set_alpha(0.0)
        ax_samples.axis('off')
        
        # Extract components
        x,y,z = samples[:,0], samples[:,1], samples[:,2]
        vpts = samples[:,-1]
        # Plot the samples
        c = plt.get_cmap(colormap)(norm(vpts))
        ax_samples.scatter(x,y,z, c=c, s=2., marker='.')
        
        # plot training signal
        tr_signal_color = (0.3, 0.3, 0.1)
        ax_tr.plot(Utr[:,0], Utr[:,1], Utr[:,2], '-', color=tr_signal_color)
        ax_tr.plot(Utr[0,0], Utr[0,1], Utr[0,2], 'k*', markersize=16)
        # If we're using the windowed method, plot every point used as an initial condition
        if train_method.key == 'windows':
            # find the dt
            dt = np.mean(tr[1:] - tr[:-1])
            # Num samples per window
            window_samples = round(params['window'] / dt)
            window_step = round((1-params['overlap']) * window_samples)
            
            ic_slice = slice(window_step, -window_samples, window_step)
            ax_tr.plot(Utr[ic_slice,0], Utr[ic_slice,1], Utr[ic_slice,2], 'ko', markersize=2, alpha=0.7)
            
        clear_3d_axis_labels(ax_tr)
        clear_3d_axis_labels(ax_samples)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=[ax_tr, ax_samples], label='Valid Prediction Time', ticks=ticks[system], shrink=0.5, fraction=0.1, anchor=(0.0,0.3))
        plt.show()
        
    except Exception as e:
        plt.close('all')
        raise

def create_plots():
    # not sure which to use
    attractor_with_train_signal('lorenz', 'windows', 1.0, 1000, 3)
