import math
from functools import partial

import tensorflow as tf
from keras.layers.core import activation
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.optimizers import Adagrad, Adam
from keras.layers.reshaping import Reshape
from keras.layers.rnn import TimeDistributed
from keras.layers.convolutional import Convolution1DTranspose, Conv1D
from keras import Sequential, Model, Input
from keras.layers import Dense, MaxPool1D, Flatten, Attention, BatchNormalization, AvgPool1D, UpSampling1D, Dropout, \
    Activation, Lambda


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)  # I use ._decayed_lr method instead of .lr

    return lr


class FFT(keras.layers.Layer):
    def __init__(self, fft_length=500):
        super(FFT, self).__init__()
        self.fft_length = tf.constant(fft_length)

    def call(self, inputs):
        return tf.signal.rfft(inputs, fft_length=self.fft_length)


class IFFT(keras.layers.Layer):
    def __init__(self, fft_length=500):
        super(IFFT, self).__init__()
        self.fft_length = tf.convert_to_tensor(fft_length, dtype=tf.int32)

    def call(self, inputs):
        return tf.signal.irfft(inputs, fft_length=self.fft_length)


class AutoEncoderModel:
    model: Model
    encoder: Model
    decoder: Model

    def __init__(self, window_size, encoded_size, dropout, num_hidden_layers=3):
        self.window_size = window_size
        self.encoded_size = encoded_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout

    def make_model(self, config):
        enc_in = Input((self.window_size,))
        x = Reshape((100, -1))(enc_in)
        sequential = Sequential()

        for i in range(self.num_hidden_layers):
            sequential.add(Dense(int(100), activation="relu"))

        x = sequential(x)
        x = Flatten()(x)
        enc_out = Dense(self.encoded_size, activation="relu")(x)

        dec_in = Dense(10000, activation="relu")(enc_out)
        x = Reshape((100, -1))(dec_in)
        sequential = Sequential()

        for i in range(self.num_hidden_layers):
            sequential.add(Dense(int(100), activation="relu"))

        x = sequential(x)
        x = Dense(100, activation="sigmoid")(x)
        x = Flatten()(x)
        dec_out = Dropout(self.dropout)(x)

        self.encoder = Model(inputs=enc_in, outputs=enc_out)
        self.decoder = Model(inputs=dec_in, outputs=dec_out)
        self.model = Model(inputs=enc_in, outputs=dec_out)
        opt = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
        self.model.compile(optimizer=opt, loss="mse", metrics=['accuracy'])

    def summary(self):
        print(self.model.summary(expand_nested=True))


class CNNAutoEncoderModel:
    model: Model
    encoder: Sequential
    decoder: Sequential

    def __init__(self, window_size, compression_size):
        self.window_size = window_size
        self.compression_size = compression_size

    def make_model(self, config):
        self.encoder = Sequential()
        self.decoder = Sequential()

        self.encoder.add(Reshape((-1, 1)))
        self.encoder.add(Conv1D(32, 3, activation='relu', padding='same', dilation_rate=2))
        self.encoder.add(MaxPool1D(2))
        self.encoder.add(Conv1D(16, 3, activation='relu', padding='same', dilation_rate=2))
        self.encoder.add(MaxPool1D(2))
        self.encoder.add(Conv1D(8, 3, activation='relu', padding='same', dilation_rate=2))
        self.encoder.add(MaxPool1D(2))
        self.encoder.add(Conv1D(4, 3, activation='relu', padding='same', dilation_rate=2))
        self.encoder.add(MaxPool1D(2))
        self.encoder.add(AvgPool1D())
        self.encoder.add(Flatten())
        self.encoder.add(Dense(2500))

        self.decoder.add(Dense(5000))
        self.decoder.add(Reshape((-1, 4)))
        self.decoder.add(Conv1D(4, 1, strides=1, activation='relu', padding='same'))
        self.decoder.add(UpSampling1D(2))
        self.decoder.add(Conv1D(8, 1, strides=1, activation='relu', padding='same'))
        self.decoder.add(UpSampling1D(2))
        self.decoder.add(Conv1D(16, 1, strides=1, activation='relu', padding='same'))
        self.decoder.add(UpSampling1D(2))
        self.decoder.add(Conv1D(32, 1, strides=1, activation='relu', padding='same'))
        self.decoder.add(UpSampling1D(2))
        self.decoder.add(Conv1D(1, 1, strides=1, activation='relu', padding='same'))
        self.decoder.add(Flatten())

        encoder_input = Input(shape=(self.window_size,))
        encoder = self.encoder(encoder_input)
        decoder = self.decoder(encoder)
        opt = Adagrad(learning_rate=config["learning_rate"],
                      initial_accumulator_value=config["initial_accumulator_value"])
        self.model = Model(inputs=encoder_input, outputs=decoder)
        self.model.compile(optimizer=opt, loss="mae", metrics=['accuracy'])

    def summary(self):
        print(self.model.summary(expand_nested=True))
