from matplotlib import pyplot as plt
import argparse
import os
import subprocess

def empty_func():
    pass

def save_figs(filename, _folder, gen_func, _extension='pdf', fn_args=tuple(), fn_kwargs=dict(), **savefig_args):
    print('{:.<30} '.format(filename), end='', flush=True)
    # Generate the plots
    # We do janky stuff to make plt.show() not do stuff; plt.ion/ioff also works but tended to mess up figure sizes
    _show = plt.show
    plt.show = empty_func
    gen_func(*args, **kwargs)
    plt.show = _show
    
    full_filename = os.path.join(_folder, filename)

    # Check how many figures exist
    fignums = plt.get_fignums()
    # Save all as needed
    if len(fignums) == 0:
        raise ValueError("Function {} did not generate any plots!".format(gen_func.__name__))
    elif len(fignums) == 1:
        plt.savefig(full_filename + '.' + _extension, format=_extension, bbox_inches='tight', **savefig_args)
    else:
        for n,i in enumerate(fignums):
            fig = plt.figure(i)
            plt.savefig(full_filename+'_'+str(n) + '.' + _extension, format=_extension, bbox_inches='tight', **savefig_args)
    plt.close('all')
    print('done')
    

if __name__ == "__main__":
    # command-line arguments
    parser = argparse.ArgumentParser(description='Generates plots for the paper.')
    _group = parser.add_mutually_exclusive_group(required=True)
    _group.add_argument('--plots', default=None, nargs='+',
                    help='Which plot to make')
    _group.add_argument('--all', action='store_true',
                    help='Generate all plots')
    _group.add_argument('--list', action='store_true',
                    help='Lists all available plots')
    parser.add_argument('--draft', action='store_true',
                    help='Create plots as .png files (faster but lower quality)')
    args = parser.parse_args()
    
    ##### Load plot data ############################################################
    # This is deferred so that argument checks/showing help/&c can be
    #   done without waiting for imports.
    
    import _vpts
    import _icmap_example
    import _attractor_with_train
    import _attractor_compare
    import _train_time_compare

    PLOT_ITEMS = [
        ('vpts', _vpts.create_vpt_plots, list(), dict()),
        ('icmap-example', _icmap_example.create_icmap_example_plot, list(), dict()),
        ('training-uniformity', _attractor_with_train.create_plots, list(), dict()),
        ('uniformity-comparison', _attractor_compare.create_plots_as_single, list(), dict(),
                    {'_extension':'png', 'dpi':500}),
        ('train-time-compare', _train_time_compare.make_plots, list(), dict()),
    ]

    
    #################################################################################
    
    if args.draft:
        folder = 'figures-draft'
    else:
        folder = 'figures'
    
    os.makedirs(folder, exist_ok=True)
    
    if args.list:
        print("Available plots:")
        print('    ' + '\n    '.join(map(lambda x:x[0], PLOT_ITEMS)))
    elif args.all:
        for filename, func, args, kwargs in PLOT_ITEMS:
            save_figs(filename, folder, func, *args, **kwargs)
        print("Done.")
    else:
        # Validate
        valid_plots = {item[0]:item for item in PLOT_ITEMS}
        invalid_args = [name for name in args.plots if name not in valid_plots]
        if len(invalid_args) > 0:
            print("Invalid plot names: '{}'".format("', '".join(invalid_args)))
        else:
            for name in args.plots:
                filename, func, args, kwargs = valid_plots[name][:4]
                if len(valid_plots[name]) == 5:
                    plt_kwargs = valid_plots[name][-1]
                else:
                    plt_kwargs = dict()
                save_figs(filename, folder, func, fn_args=args, fn_kwargs=kwargs, **plt_kwargs)
            print("Done.")