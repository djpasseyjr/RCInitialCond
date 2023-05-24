
from matplotlib import pyplot as plt
import matplotlib

from _common import *

@safeclose
def main():
    fig, ax = plt.subplots(1,1, figsize=(6,2.1))
    
    w = 0.3
    h = 0.9
    
    ax.plot([w, -w, -w, w, w], [h, h, -h, -h, h], 'k-')
    
    textparams = {
        'fontsize': 14,
        'horizontalalignment': 'center',
        'style': 'italic',
    }
    large_textparams = {
        'fontsize': 24,
        'horizontalalignment': 'center',
    }
    
    ax.text(0, h/2, "Reservoir", **large_textparams)
    ax.text(0, -h/2, "Source of\nnonlinear\ndynamics", **textparams)
    ax.text(-2.2*w, 0.3*h, "Input signal", **textparams)
    ax.text(2.2*w, 0.3*h, "Response signal", **textparams)
    
    arrowparams = dict(
        width=0.01,
        head_width=0.05,
        length_includes_head=True,
        color='k',
    )
    x1, x2 = w * 1.2, w * 3.2
    L = x2 - x1
    ax.arrow(x1, 0, L, 0, **arrowparams)
    ax.arrow(-x2, 0, L, 0, **arrowparams)
    
    ax.axis('off')
    ax.axis([-1, 1, -1, 1])
    plt.tight_layout()
    
    plt.show()