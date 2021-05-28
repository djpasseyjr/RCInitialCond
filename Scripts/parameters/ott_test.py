"""
Defines constant parameters for opt_then_test.py, with smaller parameters chosen so that the code can be tested more quickly.
"""

from numpy import exp

DATADIR = "../Data/" # Contains priors and soft robot data
BIG_ROBO_DATA = "bellows_arm1_whitened.mat"
SMALL_ROBO_DATA = "bellows_arm_whitened.mat"
VPTTOL = 0.5 # Valid prediction time error tolerance
TRAINPER = 0.66 # Percentage of the data used for training
OPT_VPT_REPS = 3
OPT_NTRIALS = 5
NSAVED_ORBITS = 5
LYAP_REPS = 5

# Time Steps for chaotic systems
DT = {
    "lorenz": 0.01,
    "rossler": 0.01,
    "thomas": 0.1
}
# Max orbit time
DURATION = {
    "lorenz": 10,
    "rossler": 40,
    "thomas": 100,
    "softrobo": 1000
}
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
    "res_sz":50,
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