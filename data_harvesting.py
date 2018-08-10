import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Stock price daily data is taken Kaggles competing on Huge stock Price dataset and the dataset is for company
filename = "aa.us.txt"


class Harvesting:
    def __init__(self):
        self.data = []
        self.mean = None
        self.std = None

    def load_data(self, filename=filename):
        date = []
        close = []
        if 'data.csv' in os.listdir():
            df = pd.read_csv('data.csv')
        # Parsing from csv to reduce the time to load data everytime
        else:
            # read data from file
            with open(filename, 'r') as f:
                for line in f.read().split()[1:]:
                    date.append(line.split(',')[0])
                    close.append((line.split(',')[4]))

            df = pd.DataFrame(data={'date': date, 'close': close})
            df.to_csv('data.csv')

        self.data = df

        return self.data

    def scale(self, timeline):
        if self.mean is None or self.std is None:
            self.mean = np.mean(np.array(self.data['close']))
            self.std = np.std(np.array(self.data['close']))

        return (timeline - self.mean) / self.std

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
                    x_i = timeseries[:-1 * predict]
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
