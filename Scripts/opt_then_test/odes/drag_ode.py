from sherpa import Continuous

def res_f(self, t, r, u, d):
    """ ODE to drive the reservoir node states with u(t) and a input signal d(t)"""
    transform_train = self.sigma * self.W_in @ u(t)
    transform_drive = self.delta * self.W_drive @ d(t)
    return -1 * self.beta * r + self.gamma * self.activ_f(self.res @ r + transform_train + transform_drive)

def res_pred_f(self, t, r, d):
    """ Reservoir prediction ode. Assumes precomputed W_out. Accepts an input signal d(t) """
    recurrence = self.sigma * self.W_in @ (self.W_out @ r)
    transform_drive =  self.delta * self.W_drive @ d(t)
    return -1 * self.beta * r + self.gamma * self.activ_f(self.res @ r + recurrence + transform_drive)
    
res_odes = {
    'res_ode':          res_f,
    'trained_res_ode':  res_pred_f,
    'opt_parameters':   [
                        Continuous(name='beta', range=[-25, 25])
                        ]
}
