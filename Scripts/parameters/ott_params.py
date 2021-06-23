"""
Defines default constant parameters for opt_then_test.py
"""

from numpy import exp

DATADIR = "../Data/" # Contains priors and soft robot data
BIG_ROBO_DATA = "bellows_arm1_whitened.mat"
SMALL_ROBO_DATA = "bellows_arm_whitened.mat"
VPTTOL = 0.5 # Valid prediction time error tolerance
TRAINPER = 0.66 # Percentage of the data used for training
OPT_VPT_REPS = 10
OPT_NTRIALS = 200
NSAVED_ORBITS = 25
LYAP_REPS = 10

# Time Steps for chaotic systems
DT = {
    "lorenz": 0.01,
    "rossler": 0.01,
    "thomas": 0.1,
    "softrobot": 0.01
}
# Max orbit time
DURATION = {
    "lorenz": 10,
    "rossler": 150,
    "thomas": 1000
}

SOFT_ROBO_TIMESTEPS = 25000

# Parameters to optimize in the reservoir computer
RES_OPT_PRMS = [
    "sigma",
    "gamma",
    "ridge_alpha",
    "spect_rad",
    "mean_degree"
]
# Window training algorithm parameters
METHOD_PRMS = [
    "window",
    "overlap"
]
# Additional Soft Robot Parameter
ROBO_OPT_PRMS = [
    "delta"
]

# Default reservoir computer hyper parameters
RES_DEFAULTS = {
    "res_sz":1000,
    "activ_f": lambda x: 1/(1+exp(-1*x)),
    "sparse_res":True,
    "uniform_weights":True,
    "signal_dim":3,
    "max_weight":2,
    "min_weight":0,
    "batchsize":2000
}

# Soft robot default parameters
ROBO_DEFAULTS = {
    "signal_dim":6,
    "drive_dim":6
}

# Prior default parameters.
# Used to make sure loaded priors aren't missing any variables.
PRIOR_DEFAULTS = {
    "sigma":0.1,
    "gamma":1.0,
    "ridge_alpha":1e-4,
    "spect_rad":0.9,
    "mean_degree":2.0,
    "window":5,
    "overlap":0.3,
    "delta":0.5
}