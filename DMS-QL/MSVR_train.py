import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from DMS.ipredictor import tools
from DMS.ipredictor.models.msvr.MSVR import *
from DMS.ipredictor.models import ANNI

if __name__ == "__main__":
    # Seed switch
    np.random.seed(36)

    time_step = 6
    gamma = 0.001
    C = 1000
    epsilon = 0.0001

    DATA = tools.data_reader('WTI.xlsx', intervals=True, resample=False)
    TRAIN, VALID, TEST = DATA[:-120], DATA[-120:], DATA[-60:]
    df = pd.read_excel('WTI.xlsx')
    data = df.iloc[2:, 1:3].values
    data = np.array(data, dtype=np.float64)
    data[:, [0, 1]] = data[:, [1, 0]]
    train, valid, test = data[:-120], data[-120:], data[-60:]

    transform_data = np.vstack((train, valid))
    num = len(transform_data)
    ss = StandardScaler()
    scaler = ss.fit(transform_data)
    X_norm = ss.transform(transform_data)

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

    msvr = MSVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon)
    # Model training
    msvr.fit(train_input, train_target)
    coefs = np.concatenate((msvr.Beta, msvr.xTrain), axis=1)

    np.savetxt('coefs_MSVR3.txt', coefs)