from matplotlib import pyplot as plt
import argparse
import os

PLOT_ITEMS = [
    ('save_name', None, list(), dict()),
]

def save_figs(filename, _folder, _extension, gen_func, *args, **kwargs):
    print('{:.<30} '.format(filename), end='')
    # Generate the plots
    plt.ion()
    gen_func(*args, **kwargs)
    plt.ioff()
    
    full_filename = os.path.join(_folder, filename)

    # Check how many figures exist
    fignums = plt.get_fignums()
    # Save all as needed
    if len(fignums) == 0:
        raise ValueError("Function {} did not generate any plots!".format(gen_func.__name__))
    elif len(fignums) == 1:
        plt.savefig(full_filename, format=_extension)
    else:
        for n,i in enumerate(fignums):
            fig = plt.figure(i)
            plt.savefig(full_filename+'_'+str(n), format=_extension)
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
    parser.add_argument('--draft', action='store_true',
                    help='Create plots as .png files (faster but lower quality)')
    args = parser.parse_args()
    
    if args.draft:
        extension = '.png'
        folder = 'figures-draft'
    else:
        extension = '.pdf'
        folder = 'figures'
    
    os.makedirs(folder, exist_ok=True)
    
    if args.all:
        for filename, func, args, kwargs in PLOT_ITEMS:
            save_figs(filename, folder, extension, func, *args, **kwargs)
        print("Done.")
    else:
        # Validate
        valid_plots = {item[0]:item for item in PLOT_ITEMS}
        invalid_args = [name for name in args.plots if name not in valid_plots]
        if len(invalid_args) > 0:
            print("Invalid plot names: '{}'".format("', '".join(invalid_args)))
        else:
            for name in args.plots:
                filename, func, args, kwargs = valid_plots[name]
                save_figs(filename, folder, extension, func, *args, **kwargs)
            print("Done.")