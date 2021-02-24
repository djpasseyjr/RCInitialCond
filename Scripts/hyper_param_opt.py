import sherpa
import rescomp as rc
import numpy as np
import sys
from measures import accduration

# Read in Arguments
system = sys.argv[1]
duration = float(sys.argv[2])
dt = float(sys.argv[3])
if len(sys.argv) > 4:
    window = sys.argv[4]
    overlap = sys.argv[5]
else:
    window = None
    overlap = 0.0

EXPERIMENT = (system, duration, dt, window, overlap)
REPS = 20
TOL = 0.2
TRAINPER = 0.85
DEFAULTS = {
    "res_sz":1000,
    "activ_f": lambda x: 1/(1+np.exp(-1*x)),
    "sparse_res":True,
    "uniform_weights":True,
    "signal_dim":3,
    "max_weight":2,
    "min_weight":0,
    "batchsize":2000
}

def train_test_orbit(system, duration=10, dt=0.01, trainper=TRAINPER):
    t, U = rc.orbit(system, duration=duration, dt=dt, trim=True)
        # Train and test data
    N = len(t)
    mid = int(N * trainper)
    tr, Utr = t[:mid], U[:mid, :]
    ts, Uts = t[mid:], U[mid:, :]
    return tr, Utr, ts, Uts

def rcompacc(system, duration, dt, window, overlap, **kwargs):
    tr, Utr, ts, Uts = train_test_orbit(system, duration=duration, dt=dt)
    rcomp = rc.ResComp(**kwargs)
    rcomp.train(tr, Utr)
    r0 = rcomp.r0
    pre = rcomp.predict(ts, r0=r0)
    idx = accduration(Uts, pre, tol=TOL)
    acc = ts[idx-1] - ts[0]
    return acc


parameters = [sherpa.Continuous(name='gamma', range=[0.1, 25], ),
              sherpa.Continuous(name='sigma', range=[0.01, 5.0]),
              sherpa.Continuous(name='spect_rad', range=[0.1, 25]),
              sherpa.Continuous(name='ridge_alpha', range=[1e-8, 2], scale='log'),
              sherpa.Continuous(name='mean_degree', range=[0.1, 5])]

algorithm = sherpa.algorithms.RandomSearch(max_num_trials=200)

study = sherpa.Study(parameters=parameters,
                 algorithm=algorithm,
                 lower_is_better=False)


for trial in study:
    for iteration in range(REPS):
        acc = rcompacc(*EXPERIMENT, **trial.parameters, **DEFAULTS)
        study.add_observation(trial=trial,
                              iteration=iteration,
                              objective=acc)
    study.finalize(trial)
    
    study.save(f"/Users/djpassey/Data/RCAlgorithm/Hyperparameter/{system}")