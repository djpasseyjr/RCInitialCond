#todo add driven part

def res_ode(self, t, r, u):
    """
    ODE imitating a spring-mass system
    """
    s, r = r[:res_sz], r[res_sz:]
    return np.concatenate((
        self.beta*-s + self.gamma * (-r + self.activ_f(self.res @ r + self.sigma * self.W_in @ u(t))),
        s
        ))

def trained_res_ode(self, t, r):
    s, r = r[:res_sz], r[res_sz:]
    return np.concatenate((
        self.beta*-s + self.gamma * (-r + self.activ_f(self.res @ r + self.sigma * self.W_in @ (self.W_out @ r))),
        s
        ))