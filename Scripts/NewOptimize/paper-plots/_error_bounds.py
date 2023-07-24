import numpy as np
import dill as pickle
from matplotlib import pyplot as plt
import rescomp as rc
from rescomp import optimizer as rcopt
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp

from _common import *

@safeclose
def create_plot(random_seed=183758375, use_local_E_D=False, E_I_cutoff=None):
    # Get parameters
    system = 'lorenz'
    windows = True
    continued = False

    augmented = 'augmented' if windows else 'standard'
    mapping = 'random' if continued else 'activ_f'
    n = 1000
    c = 1.0
    with open(f"../results/{system}-{augmented}-random-{mapping}-d{c}-n{n}.pkl", 'rb') as file:
        params = pickle.load(file)[1]
    # All of these parts are important
    params = {**rcopt.get_res_defaults(), **params}
    params['res_sz'] = n
    params['mean_degree'] = c
    params['map_initial'] = mapping

    system = rcopt.get_system(system)
    system.test_time = system.train_time * 3
    
    # Train the reservoir computer
    train_times = {
        'lorenz': 6.6,
        'rossler': 165.,
        'thomas': 660.,
    }

    np.random.seed(random_seed)
    while True:
        tr, Utr, ts, Uts = system.get_train_test_data(continued)
        res = rcopt.trained_rcomp(system, tr, Utr, **params)
        r0 = res.r0
        
        # Get a signal to test on
        if not continued:
            ts, Uts = system.get_random_test()
        ts = ts - ts[0]
        # Make the prediction
        if continued:
            pred, r = res.predict(ts, r0=r0, return_states=True)
        else:
            pred, r = res.predict(ts, u0=Uts[0], return_states=True)
        
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
        print(f"E_I: {E_I:.5f}")

        # Calculate local dynamics error
        E_D_ts = np.linalg.norm(
            res.trained_res_ode(ts, r.T).T @ res.W_out.T - system.df(ts, r @ res.W_out.T),
            axis=1
        )
        E_D = np.max(E_D_ts)
        print("E_D:", E_D)
        
        if E_I_cutoff is None or E_I < E_I_cutoff:
            break
    
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
        return L * Ebound + E_D_local(t)
        
    def bound_2_ode(t, Ebound):
        return Lambda_func(t, Uts_func(t), pred_func(t)) * Ebound + E_D_local(t)
        
    bound1 = solve_ivp(bound_1_ode, (ts[0],ts[-1]), [E_I], t_eval=ts).y[0]
    bound2 = solve_ivp(bound_2_ode, (ts[0],ts[-1]), [E_I], t_eval=ts).y[0]
    approx_bound2 = np.exp(lyapunov * ts)*(E_I + E_D/lyapunov) - E_D/lyapunov
    
    ## Plot the things
    plt.figure()
    plt.yscale('log')
    plt.plot(ts, E, 'b-', label='True error')

    plt.plot(ts, bound1, 'k:', label='Lipschitz bound (Thm 1)')
    plt.plot(ts, bound2, 'r-', label='Sharper bound (Thm 2)')
    if not use_local_E_D:
        plt.plot(ts, approx_bound2, 'g--', label=r'$e^{\lambda t}\left(E_I+\frac{E_D}{\lambda}\right)-\frac{E_D}{\lambda}$ (Approx to Thm 2)')

    plt.axis([-0.1, 5, np.min(E)*1e-3, np.max(E) * 1e3])

    plt.legend(loc='lower right')
    plt.show()
