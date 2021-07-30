
import templates

import rescomp as rc
import numpy as np
import dill as pkl

from scipy.io import loadmat
from importlib import import_module

#Maybe move to a settings file
DATADIR = "../../Data/" # Contains priors and soft robot data
BIG_ROBO_DATA = "bellows_arm1_whitened.mat"
SMALL_ROBO_DATA = "bellows_arm_whitened.mat"

def get_res_defaults():
    return {
        "res_sz":1000,
        "activ_f": lambda x: 1/(1+np.exp(-1*x)),
        "sparse_res":True,
        "uniform_weights":True,
        "max_weight":2,
        "min_weight":0,
        "batchsize":2000
    }

def get_prior_defaults():
    return {
        "sigma":0.1,
        "gamma":1.0,
        "ridge_alpha":1e-4,
        "spect_rad":0.9,
        "mean_degree":2.0,
        "window":5,
        "overlap":0.3,
        "delta":0.5
    }

##############

def load_from_file(filename, key):
    """
    Loads from a .pkl file or a .py file/module.
    If a .pkl file, returns the object.
    If a .py file, returns the object named key within that file
    """
    #Check for pickle first
    if filename[-4:]=='.pkl':
        with open(filename,'rb') as file:
            return pkl.load(file)
    #Otherwise, load it as a module
    if filename[-3:]=='.py':
        modname = filename[:-3]
    else:
        modname = filename
    try:
        mod = import_module(modname)
        return getattr(mod, key)
    except ModuleNotFoundError:
        raise FileNotFoundError(f"could not find '{filename}.'")

#####################
# Predefined systems
#####################
    
def get_system(system_name):
    """
    Gets the system with the given name.
    If system_name is one of 'lorenz', 'thomas', 'rossler', or 'softrobot', uses the predefined system object.
    Otherwise, attempts to load a file.
    
    Returns a templates.System object
    """
    
    #Numerical parameters are, in order:
    #   -train time
    #   -test time
    #   -dt
    if system_name == 'lorenz':
        return ChaosODESystem('lorenz', 30, 30, 0.01)
    elif system_name == 'rossler':
        return ChaosODESystem('rossler', 150, 150, 0.01)
    elif system_name == 'thomas':
        return ChaosODESystem('thomas', 600, 400, 0.1)
    elif system_name == 'softrobot':
        return SoftRobotSystem(150, 100, 0.01)
    else:
        try:
            return load_from_file(system_name, 'system')
        except FileNotFoundError:
            raise ValueError(f"No system found with name '{system_name}', either in built-in systems or in files.")

#Define classes to help with the default systems
class ChaosODESystem(templates.System):
    """
    Class that implements the Lorenz, Thomas, and Rossler systems.
    """
    def __init__(self, name, train_time, test_time, dt):
        if name not in {'lorenz','thomas','rossler'}:
            raise ValueError("Unsupported system type by this class")
        
        super().__init__(name, train_time, test_time, dt, signal_dim=3, is_diffeq=True, is_driven=False)
        self.df = rc.SYSTEMS[name]['df']
    
    def get_train_test_data(self, cont_test=True):
        if cont_test:
            duration = self.train_time + self.test_time
            trainper = self.train_time / duration
            tr, Utr, ts, Uts = rc.train_test_orbit(self.name, duration=duration, trainper=trainper, dt=self.dt)
        else:
            tr, Utr = rc.orbit(self.name, duration=self.train_time, trim=True)
            ts, Uts = rc.orbit(self.name, duration=self.test_time, trim=True)
        return tr, Utr, ts, Uts
    
    def get_random_test(self):
        return rc.orbit(self.name, duration=self.test_time, trim=True)
    

class SoftRobotSystem(templates.System):
    """
    Class that implements the soft robot system.
    """
    def __init__(self, train_time, test_time, dt):
        super().__init__('softrobot', train_time, test_time, dt, 6, 6, is_diffeq=False, is_driven=True)
        self.large_data = SoftRobotSystem.load_robo(DATADIR, BIG_ROBO_DATA)
        self.small_data = SoftRobotSystem.load_robo(DATADIR, SMALL_ROBO_DATA)
    
    def load_robo(datadir, filename):
        """Load soft robot data"""
        data = loadmat(datadir + filename)
        t = data['t'][0]
        q = data['q']
        pref = data["pref"]
        return t, q, pref   
        
    def get_train_test_data(self, cont_test=True):
        t, U, D = self.large_data
        
        train_steps = int(self.train_time / self.dt)
        test_steps = int(self.test_time / self.dt)
        timesteps = train_steps + test_steps
        
        t, U, D = templates.random_slice(t, U, D, timesteps)
        
        tr, ts = t[:train_steps], t[train_steps:]
        Utr, Uts = U[:train_steps, :], U[train_steps:, :]
        Dtr, Dts = D[:train_steps, :], D[train_steps:, :]
        
        if not cont_test:
            t, U, D = self.small_data
            #Make sure the slice isn't too large
            test_steps = min(test_steps,len(t))
            ts, Uts, Dts = templates.random_slice(t, U, D, test_steps)
        return tr, (Utr, Dtr), (ts, Dts), Uts
    
    def get_random_test(self):
        t, U, D = self.small_data
        
        test_steps = int(self.test_time / self.dt)
        test_steps = min(test_steps,len(t))
        
        ts, Uts, Dts = templates.random_slice(t, U, D, test_steps)
        return (ts, Dts), Uts
        











