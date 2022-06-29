import numpy as np
from scipy import linalg as la
from scipy import integrate
from scipy.interpolate import CubicSpline
import rescomp as rc
from sherpa import Continuous, Discrete

class DrivenResComp2(rc.DrivenResComp):
    """
    Version of DrivenResComp that uses a different training method, in an attempt to better handle the drive state.
    Instead of using the drive state to affect the reservoir states, it's used along the reservoir states to predict.
    """
    def __init__(self, *args, drive_alpha=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        #defaults to ridge_alpha
        if drive_alpha is None:
            self.drive_alpha = self.ridge_alpha
        else:
            self.drive_alpha = drive_alpha
        
        #Reset these to be the right size
        rd = self.res_sz * (1+self.drive_dim)
        self.Rhat = np.zeros((rd, rd))
        self.Yhat = np.zeros((rd, self.signal_dim))
    
    def res_f(self, t, r, u, d):
        """ ODE to drive the reservoir node states with u(t) and a input signal d(t) (unused here)"""
        transform_train = self.sigma * self.W_in @ u(t)
        return self.gamma * (-1 * r + self.activ_f(self.res @ r + transform_train))

    def res_pred_f(self, t, r, d):
        """ Reservoir prediction ode. Assumes precomputed W_out. Accepts an input signal d(t) """
        recurrence = self.sigma * self.W_in @ self._get_output_1d(r, d(t))
        return self.gamma*(-1*r + self.activ_f(self.res @ r + recurrence))
    
    def _get_output(self, r, d):
        """Computes Wout(r,d) for each timestep"""
        return np.einsum('tR,tD,RuD->tu', r,d, self.W_out)

    def _get_output_1d(self, r, d):
        """Computes Wout(r,d) for one timestep"""
        return np.einsum('R,D,RuD->u', r,d, self.W_out)
    
    def _augment_d(d):
        """Prepends values of 1 to the drive state d (as an array, not a CubicSpline), to improve train-ability"""
        if len(d.shape) == 1:
            return np.concatenate(((1,),d))
        else:
            tlen = d.shape[0]
            return np.hstack((np.ones((tlen,1)),d))
      
    def predict(self, t, D, u0=None, r0=None, return_states=False):
        """ Drive the reservoir node states with the training signal U and input signal D

            Parameters
            ----------
            t (1 dim array): array of time values
            D (array): for each i, D[i, :] produces the state of the input signal
                at time t[i]
            u0 (array): Initial condition of the learned system
            r0 (array): Alternately supply initial condition for reservoir nodes

            Returns
            -------
            pred (array): A (len(t) x self.res_sz) array where states[i, :] corresponds
                to the reservoir node states at time t[i]
            states (array): The state of the reservoir nodes for each time in t.
                Optional. Returned if `return_states=True`.
        """
        # Determine initial condition
        if (u0 is not None):
            r0 = self.initial_condition(u0, D[0, :])
        elif r0 is None :
            r0 = self.r0
        if not self.is_trained:
            raise Exception("Reservoir is untrained")
        #Fancy new training method
        D = DrivenResComp2._augment_d(D)
        d = CubicSpline(t, D)
        states = integrate.odeint(self.res_pred_f, r0, t, tfirst=True, args=(d,))
        pred = self._get_output(states, D)
        # Return internal states as well as predicition or not
        if return_states:
            return pred, states
        return pred
    
    def solve_wout(self):
        """ Solve the Tikhonov regularized least squares problem (Ridge regression)
            for W_out (The readout mapping)
        """
        #Check that Rhat and Yhat aren't overflowed
        if not (np.all(np.isfinite(self.Rhat)) and np.all(np.isfinite(self.Yhat))):
            raise OverflowError('overflow occurred while computing regression')
            
        #Set the regularization factors depending on whether it corresponds to an actual drive coefficient
        reg_matr = np.diag(([self.ridge_alpha]+self.drive_dim*[self.drive_alpha]) * self.res_sz)
        try:
            W_out = la.solve(self.Rhat + reg_matr, self.Yhat)
        except np.linalg.LinAlgError:
            #Try the pseudoinverse instead
            W_out = np.linalg.pinv(self.Rhat + reg_matr, Uhat) @ self.Yhat
        #Hacky things with reshaping to get it into the right dimensions
        W_out = W_out.reshape(self.res_sz, 1+self.drive_dim, -1).transpose([0,2,1])
        return W_out
    
    def update_tikhanov_factors(self, t, U, D):
        """ Drive the reservoir with the u and collect state information into
            self.Rhat and self.Yhat
            Parameters
            t (1 dim array): array of time values
            U (array): for each i, U[i, :] produces the state of the training signal
                at time t[i]
            D (array): For each i, D[i, :] produces the state of the input signal
                at time t[i]
        """
        # The i + batchsize + 1 ending adds one timestep of overlap to provide
        # the initial condition for the next batch. Overlap is removed after
        # the internal states are generated
        idxs = [(i, i + self.batchsize + 1) for i in range(0, len(t), self.batchsize)]
        # Set initial condition for reservoir nodes
        r0 = self.initial_condition(U[0, :], DrivenResComp2._augment_d(D[0, :]))
        for start, end in idxs:
            ti = t[start:end]
            Ui = U[start:end, :]
            Di = DrivenResComp2._augment_d(D[start:end, :])
            states = self.internal_state_response(ti, Ui, Di, r0)
            # Get next initial condition and trim overlap
            states, r0 = states[:-1, :], states[-1, :]
            # Update Rhat and Yhat
            rd = np.einsum('tR,tD->tRD', states, Di[:-1,:]).reshape(states.shape[0],-1)
            self.Rhat += np.einsum('tJ,tK->JK', rd, rd)
            self.Yhat += np.einsum('tJ,tu->Ju', rd, Ui[:-1,:])
        self.r0 = r0

res_odes = {
    'parameters':{'ResComp': DrivenResComp2},
    'opt_parameters':[
                Continuous(name='drive_alpha', range=[1e-8, 2], scale='log'),
                Discrete(name='res_sz', range=[50,300]
                    ],
    'remove': ['delta']
             }