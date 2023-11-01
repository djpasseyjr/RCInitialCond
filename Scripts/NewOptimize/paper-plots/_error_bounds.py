import random
import numpy as np
import dill as pickle
from matplotlib import pyplot as plt
import rescomp as rc
from rescomp import optimizer as rcopt
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp

from _common import *

def setup(random_seed, use_wa, continued=True):
    res_sz = 50
    # Get parameters
    system = 'lorenz'
    windows = False
    
    if not use_wa:
        augmented = 'augmented' if windows else 'standard'
        mapping = 'random' if continued else 'activ_f'
        n = 1000
        c = 0.1
        with open(f"../results/{system}-{augmented}-random-{mapping}-d{c}-n{n}.pkl", 'rb') as file:
            params = pickle.load(file)[1]
            
        params = {**rcopt.get_res_defaults(), **params}
        params['res_sz'] = res_sz
        params['mean_degree'] = c
        params['map_initial'] = mapping

        system = rcopt.get_system(system)
        system.train_time = 5.
        system.test_time = 5.
        
        # Set random seed
        np.random.seed(random_seed)
        r0_seed = np.random.randint(2**31)
            
        # Generate attractor orbit
        tr, Utr, ts, Uts = system.get_train_test_data(continued)
        ts -= ts[0]

        # Train res
        # Set random seed again for initial condition
        np.random.seed(r0_seed)
        res = rcopt.trained_rcomp(system, tr, Utr, **params)
        rescomp = res

        # Reobtain the training response signal
        np.random.seed(r0_seed)
        r0 = rescomp.initial_condition(Utr[0])
        r_train = rescomp.internal_state_response(tr, Utr, r0=r0)
        
        # Reconstructed signal
        train_Uhat = r_train @ rescomp.W_out.T
        
        # Make the prediction
        r0 = res.r0
        if continued:
            pred, r = res.predict(ts, r0=r0, return_states=True)
        else:
            pred, r = res.predict(ts, u0=Uts[0], return_states=True)
        
    else:
        # use_wa=True
        np.random.seed(random_seed)
        system = rcopt.get_system('lorenz')

        # Load res params
        with open(hyperparams_file('lorenz', TRAIN_METHODS['standard'], PRED_TYPES['local'], 1000, 0.1), 'rb') as file:
                 _, params = pickle.load(file)  
            
        # Generate attractor orbit
        t, U = rc.orbit('lorenz', trim=True, duration=30)
        tr_t = t[900:1100]
        tr_U = U[900:1100]

        # Train res
        res_sz = 50
        # BIG NOTE: the code Whitney has essentially was using the partial method which is technically not the standard method,
        # although it sort of also is with a really lucky randomly selected initial condtion.
        rescomp = rc.ResComp(res_sz=res_sz, mean_degree=0.1, map_initial='activ_f', **params)
        rescomp.train(tr_t, tr_U)
        res = rescomp

        # Set t, U
        r0 = rescomp.initial_condition(tr_U[0])
        res_signal = rescomp.internal_state_response(tr_t, tr_U, r0)
        r0=rescomp.r0

        # Reconstructed signal
        U_hat = res_signal @ rescomp.W_out.T

        # Predicted signal
        
        ts, Uts = rc.orbit('lorenz', initial=tr_U[-1], duration=4)
        ts -= ts[0]
        
        pred, r = rescomp.predict(ts, r0=r0, return_states=True)
    
    return res, system, ts, Uts, pred, r, r0

def search_for_params(max_E_I, max_E_D, search_seed=3935883, use_wa=True):
    rng = random.Random(search_seed)
    continued=True
    while True:
        seed = rng.randint(0, 2**31-1)
        res, system, ts, Uts, pred, r, r0 = setup(seed, use_wa, continued)
        
        u0 = Uts[0]
        if continued:
            E_I = np.linalg.norm(res.W_out @ r0 - u0)
        else:
            E_I = np.linalg.norm(res.W_out @ res.initial_condition(u0) - u0)
        
        E_D_ts = np.linalg.norm(
            res.trained_res_ode(ts, r.T).T @ res.W_out.T - system.df(ts, r @ res.W_out.T),
            axis=1
        )
        E_D = np.max(E_D_ts)
        
        print(f"{seed: <10}\t{E_I:.4f}\t{E_D:.4f}")
        
        if E_I < max_E_I and E_D < max_E_D:
            return seed


# nice seeds: 2142592748 (local=either)
# weird seeds: 110612073 (local=False), 442646541 (local=False)
@safeclose
def create_plot(random_seed=2142592748, use_local_E_D=False, E_I_cutoff=None, use_wa=True):
    
    continued=True
    
    res, system, ts, Uts, pred, r, r0 = setup(random_seed, use_wa, continued)
    
    ##################
    
    Uts_func = CubicSpline(ts, Uts)
    pred_func = CubicSpline(ts, pred)
    r_func = CubicSpline(ts, r)
    
    # Calculate true error
    E = np.linalg.norm(
        pred - Uts,
        axis=1
    )
        
    # Calculate initial error
    u0 = Uts[0]
    if continued:
        E_I = np.linalg.norm(res.W_out @ r0 - u0)
    else:
        E_I = np.linalg.norm(res.W_out @ res.initial_condition(u0) - u0)
    print(f"E_I: {E_I:.5f}    {E[0]:.5f}")

    # Calculate local dynamics error
    E_D_ts = np.linalg.norm(
        res.trained_res_ode(ts, r.T).T @ res.W_out.T - system.df(ts, r @ res.W_out.T),
        axis=1
    )
    E_D = np.max(E_D_ts)
    print("E_D:", E_D)
    
    #if E_I_cutoff is None or E_I < E_I_cutoff:
    #    break
    
    if use_local_E_D:
        E_D_local = np.vectorize(
            lambda t: np.linalg.norm(
                res.trained_res_ode(t, r_func(t).T).T @ res.W_out.T 
                - system.df(t, r_func(t) @ res.W_out.T)
            )
        )
    else:
        E_D_local = lambda t: 0*t + E_D
        
    # Lipschitz constans and Lyapunov exponents
    Dg_matrix = {
        'lorenz': lambda x,y,z: np.array([
            [-10, 10, 0],
            [28-z, -1, -x],
            [y, x, -8/3],
        ])
    }[system.name]
    L = np.max(np.vectorize(
        lambda x,y,z: np.linalg.norm(Dg_matrix(x,y,z), ord=2)
    )(*pred.T))
    print("L:",L)
    ## lyapunov exponent
    lyapunov = {
        "lorenz": 0.9056
    }[system.name]
    
    def Lambda_func(t, u1, u2):
        """
        u1, u2 are assumed to be either (3,) or (n,3) arrays.
        """
        u1 = np.array(u1).reshape(-1,3)
        u2 = np.array(u2).reshape(-1,3)
        
        diff = u1 - u2
        g_diff = system.df(t, u1) - system.df(t, u2)
        
        return np.einsum('ab,ab->a', diff, g_diff)/np.einsum('ab,ab->a', diff, diff)
    
    
    def bound_1_ode(t, Ebound):
        return L * Ebound + E_D_local(t) if Ebound < 1e10 else np.inf
        
    def bound_2_ode(t, Ebound):
        return Lambda_func(t, Uts_func(t), pred_func(t)) * Ebound + E_D_local(t)
        
    bound1 = solve_ivp(bound_1_ode, (ts[0],ts[-1]), [E_I], t_eval=ts).y[0]
    bound2 = solve_ivp(bound_2_ode, (ts[0],ts[-1]), [E_I], t_eval=ts).y[0]
    
    if use_local_E_D:
        E_D_val = np.mean(E_D_ts)
    else:
        E_D_val = E_D
    approx_bound2 = np.exp(lyapunov * ts)*(E_I + E_D_val/lyapunov) - E_D_val/lyapunov
    
    ## Plot the things
    plt.figure(figsize=(6,4))
    plt.yscale('log')
    plt.plot(ts, E, '-', label='True error', color=method_colors['windows'])

    plt.plot(ts[:len(bound1)], bound1, '-', label='Lipschitz bound (Thm 1)', color=method_colors['standard'])
    plt.plot(ts, bound2, '-', label='Lyapunov bound (Thm 2)', color=method_colors['icm'])
    plt.plot(ts, approx_bound2, '--', label=r'$e^{\lambda t}\left(E_I+\frac{E_D}{\lambda}\right)-\frac{E_D}{\lambda}$ (Approx to Thm 2)', color=(0.0, 0.3, 0.8))
    
    print(np.min(E), np.max(E))
    plt.axis([-0.1, 4, np.min(E)*2e-1, min(3e2, np.max(E) * 1e2)])

    plt.legend(loc='lower right')
    plt.show()
