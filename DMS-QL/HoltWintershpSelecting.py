import logging

from DMS.ipredictor.models import HoltWintersI
from DMS.ipredictor import tools

import numpy as np
import pandas as pd

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    data = tools.data_reader('WTI.xlsx', intervals=True, resample=False)

    # training set, validation set, testing set partition(6/2/2)
    train, valid, test = data[:-120], data[-120:-60], data[-60:]

    sp = [6, 8, 10, 12, 18, 24]  # Hyper-parameter "season_period" range setting
    result = np.zeros((len(sp), 1))

    for j in range(len(sp)):
        model = HoltWintersI(train, season_period=sp[j])
        prediction = model.predict(steps=1)

        alpha, beta, gamma = model.coefs
        alpha = alpha.A.reshape((-1))
        beta = beta.A.reshape((-1))
        gamma = gamma.A.reshape((-1))
        coefs = np.hstack((alpha, beta, gamma)).tolist()

        valid_input = data[len(train) - 1:-len(test)]
        model2 = HoltWintersI(valid_input, season_period=sp[j])
        model2.set_coefs(coefs)
        prediction2 = model2.predict(steps=1)
        model2.Xf.pop()

        prediction_df = valid.copy(deep=True)
        for i in range(len(prediction_df)):
            prediction_df.iloc[i].values = model2.Xf[i]

        result[j][0] = HoltWintersI.mape(valid, prediction_df)


    print(result)
    data = pd.DataFrame(result)

    writer = pd.ExcelWriter('HoltWintershpRaw.xlsx')
    data.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()

    writer.close