import sys
import numpy as np
import rescomp.optimizer as rcopt

def covariance(A, unbiased=True):
    """
    Computes the covariance matrix of A, where A[k,:] is the kth data point.
    unbiased=True uses the unbiased estimator, False uses the MLE
    """
    m = A.shape[0]
    if unbiased:
        m = m - 1
    Acentered = (A - np.mean(A,axis=0)) #/ np.sqrt(m)
    return np.einsum('ab,ac->bc', Acentered,Acentered) / m
    
def compute_correlation_measure(A):
    """
    Computes the sum of the squares of the correlation coefficients of the coordinates of A with each other.
    """
    m,n = A.shape
    A_cov = covariance(A)
    A_var = np.diag(A_cov)
    scaled = (A_cov**2 / A_var.reshape(1,-1))/A_var.reshape(-1,1)
    scaled[np.isnan(scaled)] = 0
    
    def _n2(k):
        return (k**2+k)/2
    #Count all intersections of 0-variance values as 1, then average. Should be in [0,1], may be slightly larger due to rounding error
    return ((np.sum(scaled)-n)/2 + _n2(np.sum(np.abs(A_var)<1e-9)))/_n2(n)


####Overwriting rescomp functions####
def update_tikhanov_factors(self, t, U):
    """ 
    Version of the function to store the correlation measure
    Drive the reservoir with the u and collect state information into
        self.Rhat and self.Yhat
        Parameters
        t (1 dim array): array of time values
        U (array): for each i, U[i, :] produces the state of the target system
            at time t[i]
    """
    # The i + batchsize + 1 ending adds one timestep of overlap to provide
    # the initial condition for the next batch. Overlap is removed after
    # the internal states are generated
    idxs = [(i, i + self.batchsize + 1) for i in range(0, len(t), self.batchsize)]
    #Prevent length-1 segment at the end
    if len(t)-idxs[-1][0] <= 1:
        idxs = idxs[:-1]
        idxs[-1] = (idxs[-1][0], len(t))
    # Set initial condition for reservoir nodes
    r0 = self.initial_condition(U[0, :])
    for start, end in idxs:
        ti = t[start:end]
        Ui = U[start:end, :]
        states = self.internal_state_response(ti, Ui, r0)
        # Get next initial condition and trim overlap
        states, r0 = states[:-1, :], states[-1, :]
        # Update Rhat and Yhat
        self.Rhat += states.T @ states
        self.Yhat += Ui[:-1, :].T @ states
        corr = compute_correlation_measure(states)
        self.correlation += corr
        print(corr)
        self.correlation_count += 1
    self.r0 = r0
    
def train(self, t, U, window=None, overlap=0):
    """ Train the reservoir computer so that it can replicate the data in U.

        Paramters
        ---------
        t (1-d array or list of 1-d arrays): Array of m equally spaced time values corresponding to signal U.
        U (array or list of arrays): Input signal array (m x self.signal_dim) where the ith row corresponds to the
            signal value at time t[i]
        window (float): If window is not `None` the reservoir computer will subdivide the input signal
            into blocks where each block corresponds to `window` seconds of time.
            Defaults to None
        overlap (float): Must be less than one and greater or equal to zero. If greater than zero, this
            will cause subdivided input signal blocks to overlap. The `overlap` variable specifies the
            percent that each signal window overlaps the previous signal window
            Defaults to 0.0
    """
    self.correlation = 0
    self.correlation_count = 0
    if isinstance(U, list) and isinstance(t, list):
        for time, signal in zip(t, U):
            idxs = self._partition(time, window, overlap=overlap)
            for start, end in idxs:
                ti = time[start:end]
                Ui = signal[start:end, :]
                self.update_tikhanov_factors(ti, Ui)
    else:
        idxs = self._partition(t, window, overlap=overlap)
        for start, end in idxs:
            ti = t[start:end]
            Ui = U[start:end, :]
            self.update_tikhanov_factors(ti, Ui)
    self.W_out = self.solve_wout()
    self.is_trained = True
    self.correlation /= self.correlation_count

def main(n):
    optimizer = rcopt.ResCompOptimizer('lorenz','activ_f','random','augmented')
    parameters = {'sigma': 2.518182559401357,
                 'gamma': 5.790760655159745,
                 'ridge_alpha': 0.0014066107907287084,
                 'spect_rad': 19.64735209662313,
                 'mean_degree': 5.0,
                 'window': 0.6711954818252962,
                 'overlap': 0.9,
                 'update_tikhanov_factors':update_tikhanov_factors,
                 'train':train}
    
    data_list = optimizer.generate_orbits(n, parameters, True)
    
    results_list = [(rcopt.get_vptime(optimizer.system, ts, Uts, pre),rcomp.correlation) for rcomp, tr, Utr, ts, Uts, pre in data_list]
           
    return results_list
    

if __name__=="__main__":
    main(1)