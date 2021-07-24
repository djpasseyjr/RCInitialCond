from soft_robot import valid_prediction_time
import numpy as np
from scipy.io import loadmat
import rescomp
import sherpa
import params as p

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

def compute_valid_prediction_time(times, U_true, U_prediction, tol=1e-3):
    root_mean_squares = nrmse(U_true, U_prediction)
    index = np.argmax(root_mean_squares > tol)
    return times[index]

if __name__=='__main__':
    time, U, D = load_robot_data(p.robot_data_path)
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

    client = sherpa.Client()
    trial = client.get_trial()

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

    num_iterations = 50
    for i in range(num_iterations):
        print(i)
        res = rescomp.DrivenResComp(**initialization_params)
        res.train(time_train, U_train, D_train, **training_params)

        U_prediction = res.predict(time_predict, D_predict)

        valid_prediction_time = compute_valid_prediction_time(time_predict, U_predict_answer, U_prediction)

        client.send_metrics(
            trial=trial, 
            iteration=i+1, 
            objective=valid_prediction_time)