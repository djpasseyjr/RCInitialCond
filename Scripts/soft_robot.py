import sherpa
import pickle
import numpy as np
from scipy.io import loadmat
from scipy import integrate
from scipy import interpolate
from os import mkdir
from matplotlib import pyplot as plt
import rescomp

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
    time, U, D = load_robot_data('/Users/taylorpool/work/webb/RCInitialCond/Data/bellows_arm_whitened.mat')
    num_times = len(time)
    dt = np.mean(time[1:num_times] - time[:num_times-1])
    train_indices = np.arange(len(time))<len(time)/2
    time_train = time[train_indices]
    U_train = U[train_indices, :]
    D_train = D[train_indices, :]
    predict_indices = ~train_indices
    time_predict = time[predict_indices]
    U_predict_answer = U[predict_indices, :]
    D_predict = D[predict_indices, :]

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

    algorithm = sherpa.algorithms.SuccessiveHalving()

    study = sherpa.Study(parameters, algorithm, False, disable_dashboard=True)

    num_iterations = 50

    for trial in study:

        initialization_params = {
            'sigma': trial.parameters['sigma'],
            'gamma': trial.parameters['gamma'],
            'ridge_alpha': trial.parameters['ridge_alpha'],
            'spect_rad': trial.parameters['spect_rad'],
            'mean_degree': trial.parameters['mean_degree'],
            'delta': trial.parameters['delta'],
            'signal_dim': 6,
            'drive_dim': 6
        }

        training_params = {
            'window': trial.parameters['window'],
            'overlap': trial.parameters['overlap']
        }

        for i in range(num_iterations):

            res = rescomp.DrivenResComp(**initialization_params)
            print(time_train.shape)
            print(U_train.shape)
            print(D_train.shape)
            res.train(time_train, U_train, D_train, **training_params)

            U_prediction = res.predict(time_predict, D_predict)

            U_train_predict = res.predict(time_train, D_train, U_train[0,:])
            study.add_observation(
                trial=trial,
                iteration=i,
                objective=valid_prediction_time(time_predict, U_predict_answer, U_prediction))
        
        study.finalize_trial()        
    

