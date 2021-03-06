import logging

from DMS.ipredictor.models import ANN, ANNI
from DMS.ipredictor import tools
from sklearn.preprocessing import StandardScaler
from DMS.ipredictor.models.msvr.MSVR import *

import numpy as np
import pandas as pd

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    file = 'WTI.xlsx'
    data = tools.data_reader('WTI.xlsx', intervals=True, resample=False)

    # training set, validation set, testing set partition(6/2/2)
    train, valid, test = data[:-120], data[-120:-60], data[-60:]
    df = pd.read_excel('WTI.xlsx')
    data_array = df.iloc[2:, 1:3].values
    data_array = np.array(data_array, dtype=np.float64)
    data_array[:, [0, 1]] = data_array[:, [1, 0]]
    train_array, valid_array, test_array = data_array[:len(train)], data_array[len(train):len(train) + len(valid)], \
                                           data_array[-len(test):]

    transform_data = np.vstack((train_array, valid_array))
    num = len(transform_data)
    ss = StandardScaler()
    scaler = ss.fit(train_array)
    X_norm = ss.transform(transform_data)

    t_s = [2, 4, 6]  # Hyper-parameter "time_step" range setting
    gamma = [0.0001, 0.001, 0.01, 0.1]  # Hyper-parameter "gamma" range setting
    C = [ 1.0, 10, 100, 1000]  # Hyper-parameter "C" range setting
    epsilon = [0.0001, 0.001, 0.01, 0.1]  # Hyper-parameter "epsilon" range setting

    writer = pd.ExcelWriter('MSVRhpRaw.xlsx')
    for m in range(len(t_s)):
        result = np.zeros((len(gamma), (len(C) * len(epsilon))))
        time_step = t_s[m]

        for l in range(len(epsilon)):
            for j in range(len(gamma)):
                for k in range((len(C))):

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
                    train_input = X[:-len(valid), :]
                    train_target = Y[:-len(valid)]
                    test_input = X[-len(valid):, :]
                    test_target = Y[-len(valid):]

                    msvr = MSVR(kernel='rbf', gamma=gamma[j], C=C[k], epsilon=epsilon[l])
                    # Model training
                    msvr.fit(train_input, train_target)
                    # Model predict on training set
                    trainPred = msvr.predict(train_input)
                    # Model testing
                    testPred = msvr.predict(test_input)

                    # reverse standardization
                    testPred = ss.inverse_transform(testPred)
                    prediction_df = valid.copy(deep=True)
                    for i in range(len(prediction_df)):
                        prediction_df.iloc[i].values = np.array([[testPred[i][0]], [testPred[i][1]]])

                    result[j][l * len(C) + k] = ANNI.mape(valid, prediction_df)
        print(result)
        output = pd.DataFrame(result)

        output.to_excel(writer, sheet_name='time_step=' + str(t_s[m]), float_format='%.5f')

    writer.save()
    writer.close

