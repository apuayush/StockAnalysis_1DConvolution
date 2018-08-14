from keras.layers import Conv1D, SeparableConv1D, MaxPool1D, Dropout, Flatten, Dense, Activation
import numpy as np
import unittest
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt


def SepConv1D(args):
    if 'input_shape' in args.keys():
        return SeparableConv1D(filters=args['filters'], kernel_size=args['kernel_size'],
                               input_shape=args['input_shape'], activation=args['activation'])
    else:
        return SeparableConv1D(filters=args['filters'], kernel_size=args['kernel_size'], activation=args['activation'])


def conv_1D(args):
    if 'input_shape' in args.keys():
        return Conv1D(filters=args['filters'], kernel_size=args['kernel_size'],
                      input_shape=args['input_shape'], activation=args['activation'])
    else:
        return Conv1D(filters=args['filters'], kernel_size=args['kernel_size'], activation=args['activation'])


def pool(args):
    return MaxPool1D(pool_size=args['pool_size'])


def dropout(args):
    return Dropout(args['ratio'])


def flatten(args):
    return Flatten()


def dense(args):
    return Dense(args['output'])


def activation(args):
    return Activation(args['function'])


class CNN():
    def __init__(self, layers):
        self.model = Sequential()
        self.history = None
        self.layers = layers
        self.layer_type = {
            'sepconv1D': SepConv1D,
            'maxpool1D': pool,
            'conv1D': conv_1D,
            'flatten': flatten,
            'dense': dense,
            'activation': activation,
            'dropout': dropout
        }

    def build_model(self):
        for layer in self.layers:
            self.model.add(self.layer_type[layer['type']](layer['args']))

        self.model.summary()

    def compile_model(self, loss="mse"):
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.95, nesterov=True)
        self.model.compile(optimizer=sgd, loss=loss, metrics=['mae'])
        pass

    def fit_model(self, X_train, Y_train, epochs=100, batch_size=32, verbose=0):
        self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=40, verbose=1, mode='auto')
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, verbose=1)
        self.history = self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.25,
                                      callbacks=[lr_reducer], shuffle=False)
        return self.history

    def predict(self, x):
        y = self.model.predict(x)
        return y

    def get_params(self):
        '''
        Method to return the parameters of the model
        :return:
        '''
        return self.model.get_config()

    def visualise_history(self):
        plt.plot(self.history.history['mean_absolute_error'])
        plt.plot(self.history.history['val_mean_absolute_error'])
        plt.title('model loss')
        plt.ylabel('mean_absolute_error')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

    def evaluate(self, X_test, Y_test):
        p = self.model.evaluate(X_test, Y_test)
        print("model loss - %f \n model mean absolute error - %f" % (p[0], p[1]))
        pre = self.model.predict(X_test)
        plt.figure(figsize=(10, 7))
        plt.plot(pre[:80])
        plt.plot(Y_test[:50])
        plt.title("Difference compared to real stock close value")
        plt.ylabel("scaled closing value")
        plt.xlabel("epoch")
        plt.legend(['predicted value', 'actual value'], loc='upper left')
        plt.show()


        # class TestCNN(TestCase):
        #     def __init__(self):
        #         pass
