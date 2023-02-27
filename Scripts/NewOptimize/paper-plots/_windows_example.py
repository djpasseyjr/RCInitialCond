from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import warnings
import dill as pickle

from _common import *

@safeclose
def create_windows_example(seed=52673):
    import rescomp as rc
    # Suppress a warning in networkx
    warnings.filterwarnings('ignore', category=FutureWarning, module='networkx')

    # Set seed for consistency
    np.random.seed(seed)
    
    with open(hyperparams_file('lorenz', TRAIN_METHODS['windows'], PRED_TYPES['global'], 1000, 0.1), 'rb') as file:
        _, params = pickle.load(file)
        
    window_size_float = params.pop('window')
    window_overlap_float = params.pop('overlap')
    display_overlap = 0.0
    
    # Generate a system trajectory from the Lorenz attractor
    t, U = rc.orbit('lorenz', trim=True, duration=30)
    n_windows = 4
    window_n_pts = int(100*window_size_float)
    window_starts = (1000 + window_n_pts*np.arange(n_windows) * (1-display_overlap)).astype(int)
    n_windows_more=2
    window_starts_more = (
        list(
            (1000 + window_n_pts*np.arange(-n_windows_more, 0) 
            * (1-display_overlap)).astype(int)
        )
        + list(
            (1000 + window_n_pts*np.arange(n_windows, n_windows+n_windows_more) 
            * (1-display_overlap)).astype(int)
        )
    )
    
    # Train a reservoir computer
    res_sz = 50
    rescomp = rc.ResComp(res_sz=res_sz, mean_degree=0.1, map_initial='activ_f', **params)
    rescomp.train(t, U, window=window_size_float, overlap=window_overlap_float)
    
    # Set everything up
    fig = plt.figure(figsize=(12,5.5))
    
    ax_sig = fig.add_axes([0.05,0.55, 0.4, 0.37])
    axs_sys = [fig.add_axes([0.05 + 0.1*j, 0.07 + 0.18*(j%2), 0.15, 0.15]) for j in range(n_windows)]
    axs_res = [fig.add_axes([0.52 + 0.1*j, 0.58 + 0.18*(j%2), 0.15, 0.15]) for j in range(n_windows)]
    ax_agg = fig.add_axes([0.55,0.08, 0.4, 0.37])
    ax_agg.set_xticks([])
    ax_agg.set_yticks([])
    
    # This one is for arrows, &c
    ax_overlay = fig.add_axes([0, 0, 1, 1])
    ax_overlay.axis([0,1,0,1])
    ax_overlay.axis('off')
    
    
    # Plot the original signal
    ax_sig.plot(
        t[900:1100 + (n_windows+1)*window_n_pts], 
        U[900:1100 + (n_windows+1)*window_n_pts], 
        color=(0.0, 0.3, 0.8)
    )
    ax_agg.plot(
        t[900:1100 + (n_windows+1)*window_n_pts], 
        U[900:1100 + (n_windows+1)*window_n_pts], 
        color=(0.0, 0.3, 0.8),
        alpha=0.5,
    )
    # Plot window boundaries
    boundaries = 1000 + window_n_pts*np.arange(2+n_windows) * (1-display_overlap)
    for idx in boundaries:
        ax_sig.axvline(
            t[int(idx)], 
            ymin=0.1, ymax=0.9,
            color='k', linestyle='--', alpha=0.8, 
        )
        ax_agg.axvline(
            t[int(idx)], 
            ymin=0.1, ymax=0.9,
            color='k', linestyle='--', alpha=0.8, 
        )
    ax_sig.axis([t[900], t[1099 + (n_windows+1)*window_n_pts], None, None])
    ax_sig.set_yticks([])
    ax_sig.set_xticks([])
    
    # Plot the window pieces and the reservoir states
    order = None
    for j, start in enumerate(window_starts):
        # Compute the starts
        tr_t = t[start:start+window_n_pts]
        tr_signal = U[start:start+window_n_pts]
        r0 = rescomp.initial_condition(tr_signal[0])
        res_signal = rescomp.internal_state_response(tr_t, tr_signal, r0)
        if order is None:
            order = np.argsort(res_signal[0])
        
        # Plot the signal
        
        ax_sys = axs_sys[j]
        ax_sys.plot(tr_t, tr_signal, color=(0.0, 0.3, 0.8))
        
        ax_sys.set_xticks([])
        ax_sys.set_yticks([])
       
        
        # Plot the reservoir response
        
        ax_res = axs_res[j]
        for i,c in enumerate(plt.get_cmap('plasma')(np.linspace(0,0.8,res_sz))):
            ax_res.plot(tr_t, res_signal[:, order[i]], '-', color=c, alpha=0.8)
        ax_res.plot([tr_t[0]]*res_sz, res_signal[0], 'k.')
        
        ax_res.spines['top'].set_visible(False)
        ax_res.spines['right'].set_visible(False)
        ax_res.spines['bottom'].set_position('zero')
        ax_res.set_xticks([])
        ax_res.set_yticks([])
        
        ax_res.axis([None,None,-1,1])
        
        # Plot the aggregated response
        ax_agg.plot(tr_t, res_signal @ rescomp.W_out.T, color=method_colors['icm'])
        
        # a r r o w s
        arrow_params = dict(
            arrowprops={
                'width': 0.002,
                'headwidth': 4,
                'headlength': 6,
                'color': 'k',
            },
            xycoords='figure fraction', 
            textcoords='figure fraction',
        )
        # Signal -> windowed signal
        ax_overlay.annotate(
            "",
            xytext=(0.125 + 0.065*j, 0.58), 
            xy=(0.05 + 0.1*j + 0.075, 
            0.07 + 0.18*(j%2) + 0.15 + 0.01),
            **arrow_params
        )
        # windowed signal -> response
        ax_overlay.annotate(
            "",
            xytext=(0.43 + 0.012*j, 0.41), 
            xy=(0.57 - 0.012*(3-j), 0.57),
            **arrow_params
        )
        # response -> aggregation
        ax_overlay.annotate(
            "",
            xy=(0.625 + 0.065*j, 0.43), 
            xytext=(0.50 + 0.1*j + 0.075, 
            0.58 + 0.18*(j%2) - 0.01),
            **arrow_params
        )
        
    # Plot more reservoir responses to get a more complete appearance
    for start in window_starts_more:
        tr_t = t[start:start+window_n_pts]
        tr_signal = U[start:start+window_n_pts]
        r0 = rescomp.initial_condition(tr_signal[0])
        res_signal = rescomp.internal_state_response(tr_t, tr_signal, r0)
        ax_agg.plot(tr_t, res_signal @ rescomp.W_out.T, color=method_colors['icm'])
        
    textparams = {
        'fontsize': 18,
        'horizontalalignment': 'center',
        'transform': ax_overlay.transAxes,
    }
    ax_overlay.text(0.25, 0.95, r'Training signal', **textparams)
    ax_overlay.text(0.25, 0.02, r'Windows', **textparams)
    ax_overlay.text(0.75, 0.95, r'Reservoir responses', **textparams)
    ax_overlay.text(0.75, 0.02, r'Aggregated responses', **textparams)
    
    ax_agg.axis([t[900], t[1099 + (n_windows+1)*window_n_pts], None, None])
    
    
    plt.show()