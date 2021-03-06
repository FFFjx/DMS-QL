import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from DMS.ipredictor.models import HoltWintersI, HybridI, LSTMI, HoltI, HoltWinters, HybridIPoints, ANN, ANNI
from DMS.ipredictor import tools

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Seed switch
    np.random.seed(36)

    # Parameters
    season_period = 24

    file = 'WTI.xlsx'
    data = tools.data_reader(file, intervals=True, resample=False)
    train, valid, test = data[:-120], data[-120:-60], data[-60:]

    model = HoltWintersI(train, season_period=season_period)
    prediction = model.predict(steps=1)

    alpha, beta, gamma = model.coefs
    alpha = alpha.A.reshape((-1))
    beta = beta.A.reshape((-1))
    gamma = gamma.A.reshape((-1))
    coefs = np.hstack((alpha, beta, gamma)).tolist()

    np.savetxt('coefs_HW2.txt', coefs)