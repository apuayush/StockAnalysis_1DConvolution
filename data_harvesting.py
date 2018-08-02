import numpy as np
import pandas as pd
import os
import json
import requests
from sklearn import preprocessing

# Stock price daily data is taken from https://www.alphavantage.co/ API from 1995 to up til now
url = 'https://www.alphavantage.co/query'
params = {
    'function': 'TIME_SERIES_DAILY',
    'symbol': 'MSFT',
    'outputsize': 'full',
    'apikey': '5COC2L9JP8SMNM41'
}


class Harvesting:
    def __init__(self):
        self.data = []

    def load_data(self, url=url, params=params):
        date = []
        close = []
        if 'data.csv' in os.listdir():
            df = pd.read_csv('data.csv')

        else:
            # get the data from api
            try:
                api_data = requests.get(url, params=params).json()
                for i in api_data['Time Series (Daily)']:
                    dt = i
                    close_value = np.float32(api_data['Time Series (Daily)'][i]['4. close'])
                    date.append(dt)
                    close.append(close_value)

            except:
                print("Network connection error or wrong url")
                return
            df = pd.DataFrame(data={'date': date, 'close': close})
            df.to_csv('data.csv', sep='\t')

        self.data = df[['date', 'close']]

        return self.data

    def split_into_chunks(self, train, predict, step, binary=False, scale=True):
        X = []
        Y = []
        for i in range(0, len(self.data), step):

            try:
                x_i = np.array(self.data.iloc[i:i + train, 1])
                y_i = self.data.iloc[i + train + predict, 1]

                if binary:
                    # for stock price hike or fall [hike, fall]
                    if y_i > 0:
                        y_i = [1., 0.]

                    else:
                        y_i = [0., 1.]
                    if scale:
                        y_i = preprocessing.scale(y_i)

                else:
                    timeseries = np.array(self.data.iloc[i:i + train + predict, 1])
                    timeseries = preprocessing.scale(timeseries)
                    x_i = timeseries[:-1]
                    y_i = timeseries[-1]

            except:
                break

            X.append(x_i)
            Y.append(y_i)

        print('total chunks ', len(X))
        print('each chunk contains', len(X[0]))
        return np.array(X), np.array(Y)

