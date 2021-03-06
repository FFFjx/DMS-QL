from DMS.ipredictor.models import HoltWintersI, HybridI, LSTMI, HoltI, HoltWinters, HybridIPoints, ANN, ANNI
from DMS.ipredictor.models.msvr.MSVR import *
from DMS.ipredictor import tools
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


def ann_predict(test_input, coefs, lookback, hidden_neurons):
    """
    :param test_input: standardized test set
    :param coefs: h5py file for model coefficient
    :param lookback: input dimension of ANN
    :param hidden_neurons: the number of neurons in hidden layer
    :return: Standardized prediction with test set, its data type is an array(len(test_input), 2)
    """
    model2 = ANNI(test_input, lookback=lookback, hidden_neurons=hidden_neurons)
    model2.set_coefs(coefs)
    pre = model2.predict(steps=1)
    model2.Xf_array = np.delete(model2.Xf_array, len(model2.Xf_array) - 1, axis=0)
    return model2.Xf_array


def lstm_predict(test_input, coefs, lookback, hidden_neurons):
    """
    :param test_input: standardized test set
    :param coefs: h5py file for model coefficient
    :param lookback: input dimension of LSTM
    :param hidden_neurons: the number of neurons in hidden layer
    :return: Standardized prediction with test set, its data type is an array(len(test_input), 2)
    """
    model2 = LSTMI(test_input, lookback=lookback, hidden_neurons=hidden_neurons)
    model2.set_coefs(coefs)
    pre = model2.predict(steps=1)
    model2.Xf_array = np.delete(model2.Xf_array, len(model2.Xf_array) - 1, axis=0)
    return model2.Xf_array


def holtwinters_predict(test_input, coeffs, season_period):
    """
    :param test_input: input for testing
    :param coeffs: txt file for model coefficient
    :param season_period: a parameter of HoltWinters model
    :return: prediction with test set, its data type is a list
    """
    coef = np.loadtxt(coeffs, dtype=np.float32)
    coefs = []
    for i in range(len(coef)):
        coefs.append(coef[i])

    model2 = HoltWintersI(test_input, season_period=season_period)
    model2.set_coefs(coefs)
    prediction2 = model2.predict(steps=1)
    model2.Xf.pop()
    return model2.Xf


def generate_hw_test_input(train, test):
    """
    :param train: training set, its data type is an array
    :param test: testing set, its data type is an array
    :return: input for testing
    """
    index_list = train.index.tolist()
    training = train.copy(deep=True)
    test_input = training.append(test)
    for i in range(len(training) - 1):
        test_input.drop(index=[index_list[i]], inplace=True)
    return test_input


def msvr_predict(test_input, coefs, gamma, C, epsilon, kernel='rbf'):
    """
    :param test_input: input for testing
    :param coefs: txt file for model coefficient
    :param gamma: a parameter of SVR model
    :param C: a parameter of SVR model
    :param epsilon: a parameter of SVR model
    :param kernel: a parameter of SVR model, use 'rbf' by default
    :return: prediction with test set, its data type is an array
    """
    msvr2 = MSVR(kernel=kernel, gamma=gamma, C=C, epsilon=epsilon)
    coef = np.loadtxt(coefs, dtype=np.float32)
    msvr2.Beta = coef[:, :2]
    msvr2.xTrain = coef[:, 2:]
    testPred = msvr2.predict(test_input)
    return testPred


def generate_msvr_test_input(train_array, test_array, length_test, time_step):
    """
    :param train_array: training set with data type array
    :param test_array: testing set with data type array
    :param length_test: length of testing data
    :param time_step: a parameter of SVR model
    :return: input for testing and standard scaler
    """
    transform_data = np.vstack((train_array, test_array))
    ss1 = StandardScaler()
    ss1.fit(train_array)
    X_norm = ss1.transform(transform_data)

    data_x, data_y = [], []
    for i in range(X_norm.shape[0] - time_step):
        if i < 0:
            continue
        x = X_norm[i:i + time_step, :]
        y = X_norm[i + time_step, :]
        x = x.reshape((-1))
        y = y.reshape((-1))
        data_x.append(x)
        data_y.append(y)

    X, Y = np.array(data_x), np.array(data_y)
    test_input = X[-length_test:, :]
    return test_input, ss1


def df_to_array(data):
    """
    :param data: input data with data type dataframe
    :return: data with data type array
    """
    X = data['values'].values.tolist()
    mixed = []
    for i in range(len(X)):
        mixed.append(X[i][0])
        mixed.append(X[i][1])
    X = np.array(mixed)
    temp = np.array(X).reshape((len(X), 1))
    return temp


def mape(real, predicted):
    """Mean absolute percentage error (MAPE) accuracy measure method
    for interval-valued data"""
    mape_h = 0
    mape_l = 0
    fitted_intervals = min(len(real), len(predicted))

    for i in range(fitted_intervals):
        mape_h += abs((real[i][0] - predicted[i][0]) / real[i][0])
        mape_l += abs((real[i][1] - predicted[i][1]) / real[i][1])
    mape_h = mape_h * 100 / fitted_intervals
    mape_l = mape_l * 100 / fitted_intervals
    return np.average([mape_h, mape_l])


def rmse(real, predicted):
    """Calculate and return rmse for interval valued data
    :param real: interval valued array
    :param predicted: predicted interval valued array
    :return: rmse result
    """
    error = 0
    for i in range(0, len(real)):
        #: difference between previous forecast value and observed value
        mean = real[i] - predicted[i]
        error += np.dot(mean.transpose(), mean)[0][0]
    return float(error)


def arv(real, predicted, sample):
    """Interval average relative variance (ARVI) accuracy measure method
    for interval-valued data
    :param sample: values for median calculation
    """
    mse_max = 0
    mse_min = 0
    mse_avg_max = 0
    mse_avg_min = 0
    sample_mean = np.mean(sample, axis=0)

    for i in range(min(len(real), len(predicted))):
        mse_max += (real[i][0] - predicted[i][0]) ** 2
        mse_min += (real[i][1] - predicted[i][1]) ** 2
        mse_avg_max += (real[i][0] - sample_mean[0]) ** 2
        mse_avg_min += (real[i][1] - sample_mean[1]) ** 2
    return float((mse_max + mse_min) / (mse_avg_max + mse_avg_min))


def test_an_agent(P, shift, action_space, Q_table, rank, mape_value, action_sequence, rank_sequence, mape_sequence):
    """
    :param P: a parameter of Q-learning. The length of Q-learning testing set
    :param shift: the current index of time step for an agent
    :param action_space: the actions which allowed to choose
    :param Q_table: Q-table
    :param rank: a matrix which recording rank of each model at each time step
    :param mape_value: a matrix which recording mape of each model at each time step
    :param action_sequence: a list which recording the agent action sequence
    :param rank_sequence: a list which recording the agent rank sequence
    :param mape_sequence: a list which recording the agent mape sequence
    :return: three lists which recording the agent action, rank and mape sequence
    """
    for p in range(P):
        if p == 0:
            # Initialize state
            perform = []
            for j in range(len(action_space)):
                perform.append(rank[action_space[j], shift-1])
            state_index = perform.index(min(perform))
            state = action_space[state_index]
            # action_sequence.append(state)
            # rank_sequence.append(rank[state, shift + p])
            # mape_sequence.append(mape_value[state, shift + p])
        else:
            state_index = last_state_index
            state = last_state
        dd = Q_table[state_index, :].tolist()
        action_index = dd.index(max(dd))
        action = action_space[action_index]
        action_sequence.append(action)
        rank_sequence.append(rank[action, shift + p])
        mape_sequence.append(mape_value[action, shift + p])
        last_state_index = action_index
        last_state = action
    return action_sequence, rank_sequence, mape_sequence


def plot_result(forecast, prediction, Q_table_record, action_sequence, best_step):
    """
    :param forecast: forecasting boundary
    :param prediction: prediction of each model
    :param Q_table_record: Q-table of each agent
    :param action_sequence: a list which recording the agent rank sequence
    :param best_step: a list which recording the best action sequence
    :return: Plotting agent learning curve and model performance
    """
    plt.figure(figsize=(14, 18))
    plt.subplot(2, 1, 1)
    for i, Q_table_i in enumerate(Q_table_record):
        plt.plot(Q_table_i, label='agent{}'.format(i))
    plt.xlabel('Episode(e)', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.legend()

    plt.subplot(2, 1, 2)
    for i, predict_result in enumerate(prediction):
        forecast1 = forecast.copy(deep=True)[R:]
        for j in range(len(forecast1)):
            forecast1.iloc[j] = predict_result.iloc[j + R][0][0]
        plt.plot(forecast1, label='model{}'.format(i + 1), linewidth=0.8)
    true_value = forecast.copy(deep=True)[R:]
    for i in range(len(true_value)):
        true_value.iloc[i] = true_value.iloc[i][0][0]
    plt.plot(true_value, label='True', color='black', linewidth=2)
    model_q = forecast[R:].copy(deep=True)
    bestValue = np.zeros((len(model_q),))
    for i in range(len(model_q)):
        model_q.iloc[i] = prediction[action_sequence[i]].iloc[i + R][0][0]
        bestValue[i] = prediction[best_step[i]].iloc[i + R][0][0]
    plt.plot(model_q, label='modelQ', color='r', linewidth=2)
    plt.scatter(forecast[R:].index, bestValue, color='b')
    plt.xlabel('Time[month]', fontsize=14)
    plt.ylabel('WTI[dollars/bbl]', fontsize=14)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # Tune the seed number if necessary. May get a better result or worse in a small scale
    np.random.seed(36)
    random.seed(20)

    file = 'WTI.xlsx'
    data = tools.data_reader(file, intervals=True, resample=False)

    # training set, validation set, testing set partition
    train, valid, test = data[:-120], data[-120:-60], data[-60:]

    # Q-learning parameters
    I = 4
    R = 36
    P = 2
    epsilon_Q = 1
    episode = 2000
    alpha = 0.01
    gamma_Q = 0.3

    # Forecasting boundary
    forecast = test

    # Transform data type from dataframe to array in order to normalization
    temp = df_to_array(data)

    # Standardization
    sscaler = StandardScaler()
    _ = sscaler.fit(temp[:2 * len(train)])
    X = sscaler.transform(temp)

    # Standardize training set, validation set and testing set with data type dataframe
    norm_X = []
    for i in range(0, len(X), 2):
        norm_X.append(X[i:i + 2, 0])
    norm_X = pd.DataFrame.from_dict({'values': norm_X})
    norm_X = norm_X.set_index(data.index)
    norm_train, norm_valid, norm_test = norm_X[:len(train)], norm_X[len(train):len(train) + len(valid)], \
                                        norm_X[-len(test):]

    # Data preprocessing for MSVR model
    df = pd.read_excel(file)
    data_array = df.iloc[2:, 1:3].values
    data_array = np.array(data_array, dtype=np.float64)
    data_array[:, [0, 1]] = data_array[:, [1, 0]]
    train_array, valid_array, test_array = data_array[:len(train)], data_array[len(train):len(train) + len(valid)], \
                                           data_array[len(train):]

    model_mape = []
    prediction = []

    # Model 1 result
    lookback = 6
    hidden_neurons = 50
    # Because of the temporality, if we want to get prediction on testing set, the input also need the last lookback
    # length of validation set
    test_input = norm_valid[-lookback:]
    test_input = test_input.append(norm_test)
    prediction2 = ann_predict(test_input, 'coefs_ANN1.h5', lookback, hidden_neurons)
    prediction2 = sscaler.inverse_transform(prediction2)
    prediction_model1 = forecast.copy(deep=True)
    for i in range(len(prediction_model1)):
        prediction_model1.iloc[i].values = np.array([[prediction2[i][0]], [prediction2[i][1]]])
    model_mape.append(ANNI.mape(forecast, prediction_model1))
    prediction.append(prediction_model1)

    # Model 2 result
    lookback = 6
    hidden_neurons = 30
    # Because of the temporality, if we want to get prediction on testing set, the input also need the last lookback
    # length of validation set
    test_input = norm_valid[-lookback:]
    test_input = test_input.append(norm_test)
    prediction2 = ann_predict(test_input, 'coefs_ANN2.h5', lookback, hidden_neurons)
    prediction2 = sscaler.inverse_transform(prediction2)
    prediction_model2 = forecast.copy(deep=True)
    for i in range(len(prediction_model2)):
        prediction_model2.iloc[i].values = np.array([[prediction2[i][0]], [prediction2[i][1]]])
    model_mape.append(ANNI.mape(forecast, prediction_model2))
    prediction.append(prediction_model2)

    # Model 3 result
    lookback = 6
    hidden_neurons = 40
    # Because of the temporality, if we want to get prediction on testing set, the input also need the last lookback
    # length of validation set
    test_input = norm_valid[-lookback:]
    test_input = test_input.append(norm_test)
    prediction2 = lstm_predict(test_input, 'coefs_LSTM1.h5', lookback, hidden_neurons)
    prediction2 = sscaler.inverse_transform(prediction2)
    prediction_model3 = forecast.copy(deep=True)
    for i in range(len(prediction_model3)):
        prediction_model3.iloc[i].values = np.array([[prediction2[i][0]], [prediction2[i][1]]])
    model_mape.append(LSTMI.mape(forecast, prediction_model3))
    prediction.append(prediction_model3)

    # Model 4 result
    lookback = 6
    hidden_neurons = 50
    # Because of the temporality, if we want to get prediction on testing set, the input also need the last lookback
    # length of validation set
    test_input = norm_valid[-lookback:]
    test_input = test_input.append(norm_test)
    prediction2 = lstm_predict(test_input, 'coefs_LSTM2.h5', lookback, hidden_neurons)
    prediction2 = sscaler.inverse_transform(prediction2)
    prediction_model4 = forecast.copy(deep=True)
    for i in range(len(prediction_model4)):
        prediction_model4.iloc[i].values = np.array([[prediction2[i][0]], [prediction2[i][1]]])
    model_mape.append(LSTMI.mape(forecast, prediction_model4))
    prediction.append(prediction_model4)

    # Model 5 result
    season_period = 10
    # Because of the temporality, if we want to get prediction on testing set, the input also need the whole
    # validation set, that's why the row 379 "valid.append(test)".
    test_input = generate_hw_test_input(train, valid.append(test))
    prediction2 = holtwinters_predict(test_input, 'coefs_HW1.txt', season_period)
    prediction_model5 = forecast.copy(deep=True)
    # And get the result of last len(test) value, that's why the row 384 "i + len(valid)"
    for i in range(len(prediction_model5)):
        prediction_model5.iloc[i].values = prediction2[i + len(valid)]
    model_mape.append(HoltWintersI.mape(forecast, prediction_model5))
    prediction.append(prediction_model5)

    # Model 6 result
    season_period = 24
    # Because of the temporality, if we want to get prediction on testing set, the input also need the whole
    # validation set, that's why the row 392 "valid.append(test)".
    test_input = generate_hw_test_input(train, valid.append(test))
    prediction2 = holtwinters_predict(test_input, 'coefs_HW2.txt', season_period)
    prediction_model6 = forecast.copy(deep=True)
    # And get the result of last len(test) value, that's why the row 397 "i + len(valid)"
    for i in range(len(prediction_model6)):
        prediction_model6.iloc[i].values = prediction2[i + len(valid)]
    model_mape.append(HoltWintersI.mape(forecast, prediction_model6))
    prediction.append(prediction_model6)

    # Model 7 result
    time_step = 2
    gamma = 0.001
    C = 1000
    epsilon = 0.0001
    test_input, ss = generate_msvr_test_input(train_array, test_array, len(test), time_step)
    prediction2 = msvr_predict(test_input, 'coefs_MSVR1.txt', gamma, C, epsilon, kernel='rbf')
    prediction2 = ss.inverse_transform(prediction2)
    prediction_model7 = forecast.copy(deep=True)
    for i in range(len(prediction_model7)):
        prediction_model7.iloc[i].values = np.array(
            [[prediction2[i][0]], [prediction2[i][1]]])
    model_mape.append(ANNI.mape(forecast, prediction_model7))
    prediction.append(prediction_model7)

    # Model 8 result
    time_step = 6
    gamma = 0.001
    C = 1000
    epsilon = 0.0001
    test_input, ss = generate_msvr_test_input(train_array, test_array, len(test), time_step)
    prediction2 = msvr_predict(test_input, 'coefs_MSVR2.txt', gamma, C, epsilon, kernel='rbf')
    prediction2 = ss.inverse_transform(prediction2)
    prediction_model8 = forecast.copy(deep=True)
    for i in range(len(prediction_model8)):
        prediction_model8.iloc[i].values = np.array(
            [[prediction2[i][0]], [prediction2[i][1]]])
    model_mape.append(ANNI.mape(forecast, prediction_model8))
    prediction.append(prediction_model8)

    print("mape for each model:", model_mape)

    # Calculate mape value of each model at each time step in data set "forecast"
    mape_value = np.zeros((len(prediction), len(forecast)))
    for i in range(mape_value.shape[1]):
        for j in range(mape_value.shape[0]):
            mape_value[j, i] = mape(forecast.iloc[i], prediction[j].iloc[i])

    # Calculate rank of each model at each time step in data set "forecast"
    rank = np.zeros((len(prediction), len(forecast)))
    for i in range(rank.shape[1]):
        grade = mape_value[:, i]
        index = np.argsort(grade)
        ranking = np.argsort(index)
        rank[:, i] = ranking + 1

    # The output of DMS
    action_sequence = []
    rank_sequence = []
    mape_sequence = []

    # Begin DMS.
    Q_table_record = []
    best_mape = []
    best_action = []
    best_rank = []

    # The number of windows equals to (len(forecast)-R)/P. Suggest to set an integer here.
    for i in range(int((len(forecast) - R) / P)):
        shift = i * P + R

        training_RL = forecast[shift - R:shift]
        test_RL = forecast[shift:shift + P]
        Q_table = np.zeros((I, I))
        epsilon1 = epsilon_Q
        mape_list = []
        for j in range(len(prediction)):
            mape_list.append(np.mean(mape_value[j, shift - R:shift]))

        # Choose the best I models among model pool
        action_space = []
        for j in range(I):
            action_index = mape_list.index(min(mape_list))
            action_space.append(action_index)
            mape_list[action_index] = 99999
        print(action_space)

        # Record the best action of each time step in Q-learning testing set
        for j in range(len(test_RL)):
            test_best = []
            for k in range(I):
                test_best.append(rank[action_space[k], shift + j])
            idx = test_best.index(min(test_best))
            best_mape.append(mape_value[action_space[idx], shift + j])
            best_action.append(action_space[idx])
            best_rank.append(rank[action_space[idx], shift + j])

        # Initialize state of Q-learning training set
        perform = []
        for j in range(I):
            perform.append(rank[action_space[j], shift - R])
        state_index = perform.index(min(perform))
        state = action_space[state_index]
        action_sequence.append(state)

        print("agent:", i)
        # To train an agent using training_RL
        temp = []
        for e in range(episode):
            action_record = []
            rank_record = []
            reward_record = []
            for t in range(len(training_RL)):
                if t == 0:
                    state = action_sequence[-1]
                    state_index = action_space.index(state)
                    action_record.append(state)
                    rank_record.append(rank[state, shift - R + t])
                else:
                    state_index = last_state_index
                    state = last_state

                # Choose an action with epsilon-greedy policy
                a = np.random.uniform()
                if a < epsilon1:
                    action_index = random.randint(0, I - 1)
                    action = action_space[action_index]
                else:
                    aa = Q_table[state_index, :].tolist()
                    action_index = aa.index(max(aa))
                    action = action_space[action_index]

                action_record.append(action)
                rank_record.append(rank[action, shift - R + t + 1])

                if t == len(training_RL) - 1:
                    action_sequence.append(action)
                    continue

                # Calculate reward. Three projects are proposed
                # reward = (rank[state, shift - R + t]) - 3*(rank[action, shift - R + t + 1]) + 8
                reward = (rank[state, shift - R + t]) - (rank[action, shift - R + t + 1])
                # reward = (rank[state, shift - R + t])**2 - (rank[action, shift - R + t + 1])**2
                reward_record.append(reward)

                # Update Q_table
                bb = Q_table[action_index, :].tolist()
                Q_table[state_index, action_index] = (1 - alpha) * Q_table[state_index, action_index] + \
                                                     alpha * (reward + gamma_Q * Q_table[
                    action_index, bb.index(max(bb))])
                last_state_index = action_index
                last_state = action

            epsilon1 = epsilon1 * 0.99
            temp.append(np.sum(Q_table))

            if e == episode - 1:
                action_sequence.pop(-2)
                # Output action, rank and reward at the last episode of each agent
                print(action_record)
                print(rank_record)
                print(reward_record)
                action_sequence.pop(-1)
                continue
            action_sequence.pop(-1)

        Q_table_record.append(temp)

        # Output Q-table of each agent after training
        print(Q_table)

        # Test an agent
        action_sequence, rank_sequence, mape_sequence = test_an_agent(P, shift, action_space, Q_table, rank, mape_value,
                                                                      action_sequence, rank_sequence, mape_sequence)
        print(action_sequence)
        print(rank_sequence)

    result_DMS = forecast[R:].copy(deep=True)
    for i in range(len(result_DMS)):
        result_DMS.iloc[i] = prediction[action_sequence[i]].iloc[R + i]
    result_best = forecast[R:].copy(deep=True)
    for i in range(len(result_DMS)):
        result_best.iloc[i] = prediction[best_action[i]].iloc[R + i]

    # Print result
    print("length of action_sequence:", len(action_sequence))
    print("length of rank_sequence:", len(rank_sequence))
    print("length of best_choice:", len(best_mape))
    print("best action sequence:", best_action)
    print("DMS action sequence:", action_sequence)
    print("best rank sequence:", best_rank)
    print("DMS rank sequence:", rank_sequence)
    print("best mape sequence:", best_mape)
    print("DMS mape sequence:", mape_sequence)
    for i in range(len(prediction)):
        print("mape of Model{}:".format(i + 1), np.mean(mape_value[i, R:]))
    print("mape of DMS:", np.mean(mape_sequence))
    print("mape of best choice:", np.mean(best_mape))
    for i in range(len(prediction)):
        print("arv of Model{}:".format(i + 1), ANNI.arv(forecast[R:], prediction[i][R:], train))
    print("arv of DMS:", ANNI.arv(forecast[R:], result_DMS, train))
    print("arv of best choice:", ANNI.arv(forecast[R:], result_best, train))

    # Plot agent learning curve and model performance if necessary
    plot_result(forecast, prediction, Q_table_record, action_sequence, best_action)