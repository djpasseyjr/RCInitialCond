
import os
import dill as pickle
import itertools

if __name__=="__main__":
    # copy parameters from the original ones
    template_orig = "results/{}-{}-d1.0-n1000.pkl"
    template_new = "traintimes/optimization/{}-{}-d1.0-n1000-tr{}.pkl"
    systems = (
        ('lorenz', 6.6),
        ('rossler', 165.0),
        ('thomas', 660.0),
    )
    augtypes = (
        'augmented-{}-activ_f',
        'standard-{}-activ_f',
        'standard-{}-random',
    )
    predtypes = (
        'continue',
        'random'
    )
    
    for (system, tr_time), augtype, predtype in itertools.product(systems, augtypes, predtypes):
        orig_filename = template_orig.format(system, augtype.format(predtype))
        new_filename = template_.format(system, augtype.format(predtype), tr_time)
        
        with open(orig_filename, 'rb') as file:
            experiment, params = pickle.load(file)
        with open(new_filename, 'wb') as file:
            pickle.dump(
                ((*experiment, tr_time), params)
                file
            )