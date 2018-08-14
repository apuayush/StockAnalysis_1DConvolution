import numpy as np
import pandas as pd
import os
import requests
from sklearn.model_selection import train_test_split
# from sklearn import preprocessing
# Stock price daily data is taken from https://www.alphavantage.co/ API from 1995 to up til now of AAPL stocks
url = 'https://www.alphavantage.co/query'
params = {
    'function': 'TIME_SERIES_DAILY',
    'symbol': 'AAPL',
    'outputsize': 'full',
    'apikey': '5COC2L9JP8SMNM41'
}


class Harvesting:
    def __init__(self):
        self.data = []
        self.mean = None
        self.std = None

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
            df.to_csv('data.csv')

        self.data = df

        return self.data

    def scale(self, timeseries):
        if self.mean is None or self.std is None:
            self.mean = np.mean(np.array(self.data['close']))
            self.std = np.std(np.array(self.data['close']))

        return (timeseries - self.mean) / self.std

    def unscale(self, Y):
        return Y * self.std + self.mean

    def split_into_chunks(self, train, predict, step, binary=False, scale=True):
        X = []
        Y = []
        for i in range(0, len(self.data), step):

            try:
                x_i = np.array(self.data['close'][i:i + train])
                y_i = self.data['close'][i + train + predict]

                if binary:
                    # for stock price hike or fall [hike, fall]
                    if y_i > 0:
                        y_i = [1., 0.]

                    else:
                        y_i = [0., 1.]

                else:
                    timeseries = np.array(self.data['close'][i:i + train + predict])
                    timeseries = self.scale(timeseries)
                    x_i = timeseries[:-predict]
                    y_i = timeseries[-predict:]

            except:
                break
            X.append(x_i)
            Y.append(y_i)

        print('total chunks ', len(X))
        print('each chunk contains', len(X[0]))
        return np.array(X), np.array(Y)

    @staticmethod
    def data_split(X, Y, split_ratio=0.1):

        def reform_data(xi):
            return xi.reshape(-1, 1)

        X = np.apply_along_axis(reform_data, 1, X)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=split_ratio)
        return X_train, X_test, Y_train, Y_test
