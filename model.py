from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Conv1D, SeparableConv1D, MaxPool1D, Dropout, Dense, Flatten, Activation
import matplotlib.pyplot as plt
import cnn


class Model:
    def __init__(self):
        self.history = None
        self.model = Sequential()

    def build_model(self, input_shape: object = (20, 1), output_shape: object = 1) -> object:
        self.model = cnn.CNN(input_shape=input_shape, layers=[
            {
                'type': 'sepconv1D',
                'args': {
                     'filters': 32,
                     'kernel_size': 5,
                     'activation': 'relu',
                    'input_shape': input_shape
                }
             },
            {
                'type': 'maxpool1D',
                'args': {
                    'pool_size': 2
                }
            },
            {
                'type': 'conv1D',
                'args': {
                    'filters': 100,
                    'kernel_size': 3,
                    'activation': 'relu'
                }
            },
            {
                'type': 'maxpool1D',
                'args': {
                    'pool_size': 2
                }
            },
            {
                'type': 'dropout',
                'args': {
                    'ratio': 0.15
                }
            },
            {
                'type': 'flatten',
                'args': None
            },
            {
                'type': 'dense',
                'args': {
                    'output': 250
                }
            },
            {
                'type': 'dropout',
                'args': {
                    'ratio': 0.2
                }
            },
            {
                'type': 'activation',
                'args': {
                    'function': 'relu'
                }
            },
            {
                'type': 'dense',
                'args': {
                    'output': output_shape
                }
            },
            {
                'type': 'activation',
                'args': {
                    'function': 'linear'
                }
            },
        ])
        # self.model.add(SeparableConv1D(filters=32, kernel_size=5, input_shape=input_shape, activation='relu'))
        # self.model.add(MaxPool1D(pool_size=2))
        # self.model.add(Conv1D(filters=100, kernel_size=3, activation='relu'))
        # self.model.add(MaxPool1D(pool_size=2))
        # self.model.add(Dropout(0.15))
        # self.model.add(Flatten())
        # self.model.add(Dense(250))
        # self.model.add(Dropout(0.20))
        # self.model.add(Activation('relu'))
        # self.model.add(Dense(output_shape))
        # self.model.add(Activation('linear'))
        # self.model.summary()
        return self.model

    def compile(self):
        sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.95, nesterov=True)
        self.model.compile(optimizer=sgd, loss='mse', metrics=['mae'])
        return self.model

    def fit(self, X_train, Y_train, epochs=300, batch_size=32):
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, verbose=1)
        self.history = self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,
                                      callbacks=[lr_reducer, early_stopper])
        return self.history

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
        plt.figure(figsize=(20,15))
        plt.plot(pre[:150])
        plt.plot(Y_test[:150])
        plt.title("Difference compared to real stock close value")
        plt.ylabel("scaled closing value")
        plt.xlabel("epoch")
        plt.legend(['predicted value', 'actual value'], loc='upper left')
        plt.show()



