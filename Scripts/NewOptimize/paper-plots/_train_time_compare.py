from matplotlib import pyplot as plt, colors, cm
import matplotlib
import dill as pickle
from glob import glob

from _common import *

train_times = {
    'lorenz': [1.0, 3.0, 6.6, 10.0, 30.0, 60.0, 100.0],
    'thomas': [10.0, 30.0, 100.0, 660.0, 1000.0, 3000.0],
    'rossler': [5.0, 15.0, 50.0, 165.0, 300.0, 1000.0, 3000.0],
}

def collect_results():

    def _get_indiv(system, pred_type, train_method, train_time):
        filename = vpts_file(system, train_method, pred_type, 1000, 1.0, traintime=train_time)
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                data = pickle.load(file)
            return np.array(data[1][pred_type.pred_type])
        else:
            return np.array([])
        
                
    results = {
        pred_type.key:{
            system:{
                train_method.key:{
                    tr_time:_get_indiv(system, pred_type, train_method, tr_time)
                    for tr_time in train_times[system]
                }
                for train_method in TRAIN_METHODS.values()
            }
            for system in SYSTEMS
        }
        for pred_type in PRED_TYPES.values()
    }
    #results[pred_type.key][system][train_method.key][train_time]
    
    return results
    
def indiv_plot(data, pred_type):
    
    colors = method_colors
    
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    
    for system, ax in zip(SYSTEMS, axs):
        for train_method in TRAIN_METHODS.values():
            times = []
            means = []
            lower_err = []
            upper_err = []
            
            for tr_time, vpts in data[pred_type.key][system][train_method.key].items():
                times.append(tr_time)
                mean = np.mean(vpts)
                mask_upper = vpts >= mean-1
                mask_lower = vpts <= mean+1
                means.append(mean)
                
                upper_err.append(
                    np.sqrt(
                        np.mean((vpts[mask_upper]-mean)**2)
                    )
                )
                lower_err.append(
                    np.sqrt(
                        np.mean((vpts[mask_lower]-mean)**2)
                    )
                )
            
            times = np.array(times)
            lower_err = np.array(lower_err)
            upper_err = np.array(upper_err)
            
            #errs = np.row_stack((lower_err, upper_err))
            
            ax.plot(
                times, means, '.', color=colors[train_method.key], 
                markersize=8.0
            )
            ax.plot(
                times, means, '-', color=colors[train_method.key], 
                alpha=1.0,
            )
            
            ax.fill_between(
                times, means - lower_err, means + upper_err,
                color=fill_colors[train_method.key],
                alpha=fill_alpha,
                edgecolor='none',
            )
            
            #ax.errorbar(
            #    times, means , yerr=errs, fmt='none', color=colors[train_method.key],
            #    capsize=4.0, capthick=2.0, alpha=0.3,
            #)
            
        ax.set_title(system.capitalize())
        ax.set_xlabel('Train time')
        ax.set_xscale('log')
        ax.set_xticks(
            train_times[system], 
            map(
                lambda x:str(int(x)) if x != 6.6 else str(x), 
                train_times[system]
            )
        )
        ax.tick_params(axis='x', which='minor', color='none')
        
        ax.axis([None,None,0,None])
    
    plt.suptitle("{} accuracy (VPT)".format(pred_type.name))
    axs[0].set_ylabel("VPT")
    legend_items = [
        matplotlib.lines.Line2D([0],[0], color=colors[tr_key], lw = 4, label=train_method.name)
        for (tr_key, train_method) in TRAIN_METHODS.items()
    ]
    axs[-1].legend(handles=legend_items, loc=(0.05, 0.75), fontsize=10.0, framealpha=1.0)
    
    
@safeclose
def make_plots():
    data = collect_results()
    
    for pred_type in PRED_TYPES.values():
        indiv_plot(data, pred_type)
    plt.show()