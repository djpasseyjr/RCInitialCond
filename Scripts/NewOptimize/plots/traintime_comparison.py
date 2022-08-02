import numpy as np
from matplotlib import pyplot as plt


def create_plot(data, **plotparams):
    means = []
    upper = []
    lower = []
    
    for item in data:
        mean = np.mean(item)
        mask = item < mean
        upper.append(np.sqrt(np.mean((item[~mask]-mean)**2)))
        lower.append(np.sqrt(np.mean((item[mask]-mean)**2)))
    
    plt.errorbar()
