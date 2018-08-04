from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv1D, SeparableConv1D, MaxPool1D, Dropout, Dense, Flatten, Activation
import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        self.history = None
        self.model = Sequential()

    def build_model(self, input_shape=(20, 1), output_shape=1 ):
        self.model.add(SeparableConv1D(filters=32, kernel_size=5, input_shape=input_shape, activation='relu'))
        self.model.add(MaxPool1D(pool_size=2))
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
        self.model.add(MaxPool1D(pool_size=2))
        self.model.add(Dropout(0.15))
        self.model.add(Flatten())
        self.model.add(Dense(250))
        self.model.add(Dropout(0.20))
        self.model.add(Activation('relu'))
        self.model.add(Dense(output_shape))
        self.model.add(Activation('linear'))
        self.model.summary()
        return self.model

    def compile(self):
        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.95, nesterov=True)
        self.model.compile(optimizer=sgd, loss='mse', metrics=['mae'])
        return self.model

    def fit(self, X_train, Y_train, epochs=300, batch_size=32):
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, verbose=1)
        self.history = self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,callbacks=[lr_reducer])
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

