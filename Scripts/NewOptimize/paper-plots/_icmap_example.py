from matplotlib import pyplot as plt
import matplotlib
import rescomp as rc
import numpy as np
import warnings
import sys

from _common import *

def create_icmap_example_plot(seed=73323):

    # fix the seed for consistency
    np.random.seed(seed)
    # ignore a FutureWarning in rescomp
    warnings.filterwarnings('ignore', category=FutureWarning, module='networkx')
    
    # Generate a system trajectory from the Lorenz attractor
    t, U = rc.orbit('lorenz', trim=True, duration=30)
    num_pts=300
    tr_t = t[1000:1000+num_pts]
    tr_signal = U[1000:1000+num_pts]
    
    # Create a reservoir computer
    res_sz = 50
    rescomp = rc.ResComp(res_sz=res_sz, mean_degree=0.1, map_initial='activ_f')
    r0 = rescomp.initial_condition(tr_signal[0])
    res_signal = rescomp.internal_state_response(tr_t, tr_signal, r0)
    
    try:
        # figure time
        fig = plt.figure(figsize=(12,6))
        ax_sys = fig.add_subplot(1,2,1,projection='3d')
        ax_res = fig.add_subplot(1,2,2)
        
        # This one is for arrows, &c
        ax_overlay = fig.add_subplot(1,1,1)
        ax_overlay.axis([0,1,0,1])
        ax_overlay.axis('off')
        
        # Plot the trajectory
        train_split = num_pts - 10
        ax_sys.plot(U[:,0], U[:,1], U[:,2], '-', alpha=0.7, color=(0.0, 0.3, 0.8))
        train_color = (0.3, 0.3, 0.1)
        ax_sys.plot(tr_signal[:train_split,0], tr_signal[:train_split,1], tr_signal[:train_split,2], '-', color=train_color)
        ax_sys.plot(tr_signal[train_split:,0], tr_signal[train_split:,1], tr_signal[train_split:,2], '--', color=train_color)
        ax_sys.plot(tr_signal[0,0], tr_signal[0,1], tr_signal[0,2], 'k*', markersize=12.)
        
        ax_sys.set_xticks(np.linspace(-20,20,5), labels=[])
        ax_sys.set_yticks(np.linspace(-25,30,5), labels=[])
        ax_sys.set_zticks(np.linspace(5,45,5), labels=[])
        #ax_sys.axis('off')
        
        # Plot reservoir states
        order = np.argsort(res_signal[0])
        for i,c in enumerate(plt.get_cmap('plasma')(np.linspace(0,0.8,res_sz))):
            ax_res.plot(tr_t, res_signal[:, order[i]], '-', color=c, alpha=0.8)
        ax_res.plot([tr_t[0]]*res_sz, res_signal[0], 'k.')
        
        ax_res.spines['top'].set_visible(False)
        ax_res.spines['right'].set_visible(False)
        ax_res.spines['bottom'].set_position('zero')
        ax_res.set_xticks([])
        ax_res.set_yticks([-1,0,1])
        
        # Label for initial conditions
        textparams = {
            'fontsize': 16,
            'horizontalalignment': 'center',
        }
        ax_sys.text(tr_signal[0,0], tr_signal[0,1], tr_signal[0,2]+5, r"$\mathbf{u}_0$", **textparams)
        ax_res.text(tr_t[0], 1.0, r"$\mathbf{r}_0$", **textparams)
        
        bounding_box = ax_res.get_position()
        height = 0.5
        bounding_box.y0 = 0.5 - height/2
        bounding_box.y1 = 0.5 + height/2
        ax_res.set_position(bounding_box)
        
        # temporary; show bounds of the overlay subplot
        #ax_overlay.plot([0,0,1,1,0],[0,1,1,0,0], 'k:')
        #ax_overlay.plot([0,1],[1,0], 'k:')
        #ax_overlay.plot([0,1],[0,1], 'k:')
        
        # Arrow showing mappings
        path_t = np.linspace(-1,1,3)
        path_x = 0.5 + path_t * 0.15
        path_y = 0.15*(1-path_t**2)
        codes = [
            matplotlib.path.Path.MOVETO
        ] + [
            matplotlib.path.Path.CURVE3
        ] * (len(path_t) - 1)
        
        path1 = matplotlib.path.Path(np.column_stack((path_x, 0.85+path_y)), codes=codes)
        path2 = matplotlib.path.Path(np.column_stack((path_x, 0.18-path_y)), codes=codes)
        
        ax_overlay.add_patch(matplotlib.patches.FancyArrowPatch(path=path1, arrowstyle="->", mutation_scale=20.))
        ax_overlay.add_patch(matplotlib.patches.FancyArrowPatch(path=path2, arrowstyle="<-", mutation_scale=20.))
        textparams = {
            'fontsize': 18,
            'horizontalalignment': 'center',
            'transform': ax_overlay.transAxes,
        }
        ax_overlay.text(0.5, 0.95, r'$\Phi$', **textparams)
        ax_overlay.text(0.5, 0.05, r'$W_{\mathrm{out}}\approx \Phi^{-1}$', **textparams)
        
        
        plt.show()
    except Exception as e:
        # cleanup
        plt.close('all')
        raise
        