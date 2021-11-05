import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from DMS.ipredictor.models import HoltWintersI, HybridI, LSTMI, HoltI, HoltWinters, HybridIPoints, ANN, ANNI
from DMS.ipredictor import tools

if __name__ == "__main__":
    # Seed switch
    np.random.seed(36)

    file = 'WTI.xlsx'
    data = tools.data_reader(file, intervals=True, resample=False)
    train, valid, test = data[:-120], data[-120:-60], data[-60:]

    # ANN parameters
    lookback = 6
    hidden_neurons = 50

    X = data['values'].values.tolist()
    mixed = []
    for i in range(len(X)):
        mixed.append(X[i][0])
        mixed.append(X[i][1])
    X = np.array(mixed)
    temp = np.array(X).reshape((len(X), 1))

    # Standardize training set
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

    model = LSTMI(norm_train, lookback=lookback, hidden_neurons=hidden_neurons)
    prediction1 = model.predict(steps=1)  # Training
    model.save_coefs('coefs_LSTM3.h5')