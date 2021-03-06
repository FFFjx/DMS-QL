import logging

from DMS.ipredictor.models import ANN, ANNI, LSTMI
from DMS.ipredictor import tools
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    data = tools.data_reader('WTI.xlsx', intervals=True, resample=False)

    # training set, validation set, testing set partition(6/2/2)
    train, valid, test = data[:-120], data[-120:-60], data[-60:]

    X = data['values'].values.tolist()
    mixed = []
    for i in range(len(X)):
        mixed.append(X[i][0])
        mixed.append(X[i][1])
    X = np.array(mixed)
    temp = np.array(X).reshape((len(X), 1))

    # Standardization
    scaler = StandardScaler()
    _ = scaler.fit(temp[:2 * len(train)])
    X = scaler.transform(temp)

    # Standardized training set, validation set and testing set with data type dataframe
    norm_X = []
    for i in range(0, len(X), 2):
        norm_X.append(X[i:i + 2, 0])
    norm_X = pd.DataFrame.from_dict({'values': norm_X})
    norm_X = norm_X.set_index(data.index)
    norm_train, norm_valid, norm_test = norm_X[:len(train)], norm_X[len(train):len(train) + len(valid)], \
                                        norm_X[-len(test):]

    lb = [6, 12, 24]  # Hyper-parameter "lookback" range setting
    h_n = [10, 20, 30, 40, 50]  # Hyper-parameter "hidden_neurons" range setting
    seed = [8, 16, 23, 2, 25]  # Seed setting, used for experiment replication
    result = np.zeros((len(h_n), (len(lb)*len(seed))))

    for l in range(len(seed)):
        np.random.seed(seed[l])
        for j in range(len(h_n)):
            for k in range((len(lb))):
                lookback = lb[k]
                model = LSTMI(norm_train, lookback=lookback, hidden_neurons=h_n[j])
                prediction1 = model.predict(steps=1)  # Training

                testX = norm_train[-lookback:]
                testX = testX.append(norm_valid)
                # Transform dataframe to array
                testX = testX['values'].values.tolist()
                mixed = []
                for i in range(len(testX)):
                    mixed.append(testX[i][0])
                    mixed.append(testX[i][1])
                testX = np.array(mixed)
                testX = np.array(testX).reshape((len(testX), 1))
                testingX = []
                for i in range(0, len(testX) - lookback * 2, 2):
                    shift = i + lookback * 2
                    testingX.append(testX[i:shift, 0])
                testingX = np.array(testingX)
                # LSTM need to reshape input
                shape = testingX.shape
                testingX = np.reshape(testingX, (shape[0], 1, shape[1]))
                # Input testingX must be an array
                prediction = model.model.predict(testingX)  # Testing
                prediction = scaler.inverse_transform(prediction)
                prediction_df = valid.copy(deep=True)
                for i in range(len(prediction_df)):
                    prediction_df.iloc[i].values = np.array([[prediction[i][0]], [prediction[i][1]]])

                result[j][l*len(lb)+k] = LSTMI.mape(valid, prediction_df)
    print(result)
    data = pd.DataFrame(result)

    writer = pd.ExcelWriter('LSTMhpRaw.xlsx')
    data.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()

    writer.close