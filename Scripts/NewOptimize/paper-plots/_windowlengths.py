
import numpy as np
import rescomp as rc
from matplotlib import pyplot as plt

from _common import *

@safeclose
def create_plot(system, dt=0.01, L_background=100, L=1.0, n_orbits=6, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')
        show=True
    else:
        show=False
    
    t,U = rc.orbit(system, trim=True, duration=L_background, dt=dt)
    
    #back_color, alpha = 'b', 0.3
    back_color, alpha = (0.5, 0.7, 0.8),0.7
    
    ax.plot(*U.T, color=back_color, alpha=alpha)
    if system=='thomas':
        ax.plot(*(-U.T), color=back_color, alpha=alpha)
    
    for i in range(n_orbits):
        if system=='thomas':
            ic = np.random.random(3) * 100 - 50
        else:
            ic = None
        t,U = rc.orbit(system, trim=True, duration=L, dt=dt, initial=ic)
        
        #color = f'C{i}'
        color = (0.0, 0.3, 0.8)
        
        ax.plot(*U.T, color=color)
        ax.plot(*U[0], color=color, marker='.')
    
    ax.axis('off')
    if show:
        plt.show()

@safeclose
def all_plots():
    fig = plt.figure()
    ax1 = fig.add_axes([-0.05, 0.25, 0.6, 0.7], projection='3d', facecolor=(0,0,0,0))
    ax2 = fig.add_axes([0.45, 0.40, 0.6, 0.7], projection='3d', facecolor=(0,0,0,0))
    ax3 = fig.add_axes([0.2, -0.05, 0.6, 0.7], projection='3d', facecolor=(0,0,0,0))
    
    create_plot('lorenz', ax=ax1, L_background=50)
    create_plot('rossler', L=3, L_background=500, ax=ax2)
    create_plot('thomas', L=10, L_background=1600, ax=ax3)
    
    plt.show()

def all_plots_flat(seed1=1247824, seed2=1936402, seed3=2498284):
    fig = plt.figure(figsize=(12,2.5))
    w = 0.8
    h = 1.5
    ax1 = fig.add_axes([1/6-w/2, -.25, w, h], projection='3d', facecolor=(0,0,0,0))
    ax2 = fig.add_axes([1/2-w/2, -.15, w, h], projection='3d', facecolor=(0,0,0,0))
    ax3 = fig.add_axes([5/6-w/2, -.25, w, h], projection='3d', facecolor=(0,0,0,0))
    
    np.random.seed(seed1)
    create_plot('lorenz', ax=ax1, L_background=150)
    np.random.seed(seed2) #1936393
    create_plot('rossler', L=3, L_background=500, ax=ax2)
    np.random.seed(seed3)
    create_plot('thomas', L=10, L_background=1600, ax=ax3)
    
    plt.show()