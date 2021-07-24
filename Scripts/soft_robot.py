import sherpa
import pickle
import numpy as np
from os import mkdir
from scipy.io import loadmat
from matplotlib import pyplot as plt
import rescomp
import sys
import params as p
import os

class ReservoirParams:
    def __init__(self, U):
        self.n_r = 10
        self.n_s = U.shape[1]
        self.gamma = 1.0
        self.f = np.tanh
        self.sigma = 1.0
        self.W_in = np.random.rand(self.n_r, self.n_s)
        self.A = np.random.rand(self.n_r, self.n_r)

def load_robot_data(filepath):
    data = loadmat(filepath)
    t = data['t'][0]
    q = data['q']
    pressure = data['pref']
    return t, q, pressure

def nrmse(true, pred):
    """ Normalized root mean square error. (A metric for measuring difference in orbits)
    Parameters:
        Two mxn arrays. Axis zero is assumed to be the time axis (i.e. there are m time steps)
    Returns:
        err (ndarray): Error at each time value. 1D array with m entries
    """
    sig = np.std(true, axis=0)
    err = np.linalg.norm((true-pred) / sig, axis=1, ord=2)
    return err

def valid_prediction_time(times, U_true, U_prediction, tol=1e-3):
    root_mean_squares = nrmse(U_true, U_prediction)
    index = np.argmax(root_mean_squares > tol)
    return times[index]

def plot_performance(res):
    plt.figure()
    plt.plot(time, U, label=['response0', 'response1', 'response2', 'response3', 'response4', 'response5'])
    plt.title('Soft Robot System Response')
    plt.legend()

    plt.figure()
    plt.title('Soft Robot Pressure Inputs')
    plt.plot(time, D, label=['input0', 'input1', 'input2', 'input3', 'input4', 'input5'])
    plt.legend()

    plt.figure()
    plt.subplot(3,2,1)
    plt.title('Prediction 0')
    plt.plot(time_train, U_train[:, 0], '--', label='answer')
    plt.plot(time_train, U_train_predict[:, 0], '-', label='before prediction')
    plt.plot(time_predict, U_predict_answer[:, 0], '--', label='answer')
    plt.plot(time_predict, U_prediction[:, 0], '-', label='prediction')

    plt.subplot(3,2,2)
    plt.title('Prediction 1')
    plt.plot(time_predict, U_predict_answer[:, 1], '--', label='answer')
    plt.plot(time_train, U_train[:, 1], '--', label='answer')
    plt.plot(time_train, U_train_predict[:, 1], '-', label='before prediction')
    plt.plot(time_predict, U_prediction[:, 1], '-', label='prediction')

    plt.subplot(3,2,3)
    plt.title('Prediction 2')
    plt.plot(time_train, U_train[:, 2], '--', label='answer')
    plt.plot(time_train, U_train_predict[:, 2], '-', label='before prediction')
    plt.plot(time_predict, U_predict_answer[:, 2], '--', label='answer')
    plt.plot(time_predict, U_prediction[:, 2], '-', label='prediction')

    plt.subplot(3,2,4)
    plt.title('Prediction 3')
    plt.plot(time_train, U_train[:, 3], '--', label='answer')
    plt.plot(time_train, U_train_predict[:, 3], '-', label='before prediction')
    plt.plot(time_predict, U_predict_answer[:, 3], '--', label='answer')
    plt.plot(time_predict, U_prediction[:, 3], '-', label='prediction')

    plt.subplot(3,2,5)
    plt.title('Prediction 4')
    plt.plot(time_train, U_train[:, 4], '--', label='answer')
    plt.plot(time_train, U_train_predict[:, 4], '-', label='before prediction')
    plt.plot(time_predict, U_predict_answer[:, 4], '--', label='answer')
    plt.plot(time_predict, U_prediction[:, 4], '-', label='prediction')

    plt.subplot(3,2,6)
    plt.plot(time_train, U_train[:, 5], '--', label='answer')
    plt.plot(time_train, U_train_predict[:, 5], '-', label='before prediction')
    plt.title('Prediction 5')
    plt.plot(time_predict, U_predict_answer[:, 5], '--', label='answer')
    plt.plot(time_predict, U_prediction[:, 5], '-', label='prediction')

    plt.show()

if __name__=='__main__':
    time, U, D = load_robot_data(p.robot_data_path)
    num_times = len(time)
    dt = np.mean(time[1:num_times] - time[:num_times-1])
    parameters = [
        sherpa.Continuous('sigma', [0.01, 5.0]),
        sherpa.Continuous('gamma', [0.1, 25]),
        sherpa.Continuous('ridge_alpha', [1e-8, 2], scale='log'),
        sherpa.Continuous('spect_rad', [0.1, 25]),
        sherpa.Continuous('mean_degree', [0.1, 5]),
        sherpa.Continuous('window', [dt*10, dt*1000]),
        sherpa.Continuous('overlap', [0.0, 0.95]),
        sherpa.Continuous('delta', [0.01, 5.0]),
    ]

    algorithm = sherpa.algorithms.RandomSearch()

    scheduler = sherpa.schedulers.SLURMScheduler(
        submit_options="-N soft_robot_trial -P soft_robot_project -q soft_robot_queue",
        environment=os.path.join(p.rc_cond_path, 'Slurm/environment_setup.sh')
    )

    results = sherpa.optimize(
        parameters=parameters,
        algorithm=algorithm,
        lower_is_better=True,
        scheduler=scheduler,
        filename='run_trial.py',
        disable_dashboard=True
    )