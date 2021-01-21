import rescomp as rc
import numpy as np
import pickle as pkl
import sys

DEFAULTHYP = {
    "res_sz" : 1000,
    "activ_f" : lambda x: 1/(1 + np.exp(-1*x))
}

SYSHYP = {
    "lorenz" : {
        "gamma" : 19.1,
        "mean_degree" : 2.0,
        "ridge_alpha" : 6e-7,
        "sigma" : 0.063,
        "spect_rad" : 8.472
    },
    "rossler" : {
        "gamma" : 5.632587,
        "mean_degree" : 0.21,
        "ridge_alpha" : 2e-7,
        "sigma" : 0.078,
        "spect_rad" : 14.6
    },
    "thomas" : {
        "gamma" : 12.6,
        "mean_degree" : 2.2,
        "ridge_alpha" : 5e-4,
        "sigma" : 1.5,
        "spect_rad" : 12.0
    }
}

ORBITHYP = {
    "lorenz" : {
        "duration" : 60,
        "trainper" : 0.66,
        "dt": 0.01,
    },
    "rossler" : {
        "duration" : 150,
        "trainper" : 0.5,
        "dt": 0.01,
    },
    "thomas" : {
        "duration" : 1000,
        "trainper" : 0.9,
        "dt" : 0.1
    }
}

N = 200
NLYAP = 10
OVERLAP = 0.95
system = sys.argv[1]
algo = sys.argv[2]

results = {name:[] for name in ["continue_r0", "rand_u0", "cont_sys_err", "rand_sys_err", "lyapunov"]}

for i in range(N):
    tr, Utr, ts, Uts = rc.train_test_orbit(system, **ORBITHYP[system])
    rcomp = rc.ResComp(**DEFAULTHYP, **SYSHYP[system])
    if algo =="old":
        rcomp.train(tr, Utr)
    else:
        window = ORBITHYP[system]["duration"] / 10.0
        rcomp.train(tr, Utr, window=window, overlap=OVERLAP)
    # Continued r0
    r0 = rcomp.r0
    pre = rcomp.predict(ts, r0=r0)
    i = rc.accduration(Uts, pre, order=2)
    accdur = ts[i] - ts[0]
    results["continue_r0"].append(accdur)
    err = rc.system_fit_error(ts, pre, system)
    trueerr = rc.system_fit_error(ts, Uts, system)
    results["cont_sys_err"].append((trueerr, err))
    # Random r0
    tr, Utr, ts, Uts = rc.train_test_orbit(system, **ORBITHYP[system])
    u0 = Uts[0, :]
    randpre = rcomp.predict(ts, u0)
    randi = rc.accduration(Uts, randpre, order=2)
    randaccdur = ts[randi] - ts[0]
    results["rand_u0"].append(randaccdur)
    randerr = rc.system_fit_error(ts, randpre, system, order="inf")
    trueranderr = rc.system_fit_error(ts, Uts, system, order="inf")
    results["rand_sys_err"].append((trueranderr, randerr))
    # Lyapunov
    lam = 0
    for i in range(NLYAP):
        delta0 = np.random.randn(DEFAULTHYP["res_sz"]) * 1e-6
        predelta = rcomp.predict(ts, r0=r0+delta0)
        i = rc.accduration(pre, predelta)
        lam += rc.lyapunov(ts[:i], pre[:i, :], predelta[:i, :], delta0)
    results["lyapunov"].append(lam / NLYAP)

f = open(f"{system}_{algo}.pkl", "wb")
pkl.dump(results, f)
