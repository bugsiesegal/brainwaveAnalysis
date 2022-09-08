import math

import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adagrad, Adam
from keras.layers.reshaping import Reshape
from keras.layers.rnn import TimeDistributed
from keras.layers.convolutional import Convolution1DTranspose, Conv1D
from keras import Sequential, Model, Input
from keras.layers import Dense, MaxPool1D, Flatten, Attention, BatchNormalization


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)  # I use ._decayed_lr method instead of .lr

    return lr


class AutoEncoderModel:
    model: Model
    encoder: Model
    decoder: Model

    def __init__(self, window_size, encoded_size, num_hidden_layers=2):
        self.window_size = window_size
        self.encoded_size = encoded_size
        self.num_hidden_layers = num_hidden_layers

    def make_model(self, config):
        enc_in = Input((self.window_size,))
        x = Sequential([
            Dense(self.window_size * 8) for i in range(self.num_hidden_layers)
        ])(enc_in)
        enc_out = Dense(self.encoded_size, activation="sigmoid")(x)
        dec_in = Dense(self.encoded_size)(enc_out)
        x = Sequential([
            Dense(self.window_size * 8) for i in range(self.num_hidden_layers)
        ])(dec_in)
        dec_out = Dense(self.window_size, activation="sigmoid")(x)

        self.encoder = Model(inputs=enc_in, outputs=enc_out)
        self.decoder = Model(inputs=dec_in, outputs=dec_out)
        self.model = Model(inputs=enc_in, outputs=dec_out)
        opt = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
        self.model.compile(optimizer=opt, loss="mse", metrics=['accuracy'])

    def summary(self):
        print(self.model.summary(expand_nested=False))


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

        self.encoder.add(Reshape((100, int(self.window_size / 100),), input_shape=(self.window_size,)))
        self.encoder.add(Dense(int(self.window_size / 100), activation="linear"))
        self.encoder.add(Conv1D(2 * int(self.window_size / 100), 10, padding="same", activation="linear"))
        self.encoder.add(Conv1D(1 * int(self.window_size / 100), 10, padding="same", activation="linear"))
        self.encoder.add(Conv1D(10, 10, padding="same", activation="linear"))
        self.encoder.add(Conv1D(1, 10, padding="same", activation="linear"))
        self.encoder.add(Flatten())
        self.encoder.add(Dense(self.compression_size, activation="linear"))

        self.decoder.add(Dense(self.compression_size, input_shape=(self.compression_size,), activation="linear"))
        self.decoder.add(Reshape((self.compression_size, 1)))
        self.decoder.add(Convolution1DTranspose(1, 1, padding="same", activation="linear"))
        self.decoder.add(Convolution1DTranspose(10, 1, padding="same", activation="linear"))
        self.decoder.add(Convolution1DTranspose(1 * int(self.window_size / 100), 1, padding="same", activation="linear"))

        self.decoder.add(Reshape((int(self.window_size / 100), 100,)))
        self.decoder.add(Dense(100, activation="linear"))
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


class TestModel:
    model: Model
    encoder: Sequential
    decoder: Sequential

    def __init__(self, window_size, compression_size):
        self.window_size = window_size
        self.compression_size = compression_size

    def make_model(self, config):
        self.encoder = Sequential()
        self.decoder = Sequential()

        self.encoder.add(Dense(self.window_size, input_shape=(100,), activation="linear"))
        self.encoder.add(Dense(self.window_size, activation="linear"))
        self.encoder.add(Reshape((1, self.window_size)))
        self.encoder.add(Conv1D(2000, 1, activation="linear"))
        self.encoder.add(Conv1D(1000, 1, activation="linear"))
        self.encoder.add(Conv1D(10, 1, activation="linear"))
        self.encoder.add(Flatten())
        self.encoder.add(Dense(self.compression_size, activation="linear"))
        self.encoder.add(Dense(self.compression_size, activation="linear"))

        self.decoder.add(Dense(self.compression_size, input_shape=(self.compression_size,), activation="linear"))
        self.decoder.add(Dense(self.compression_size, activation="linear"))
        self.decoder.add(Reshape((1, self.compression_size)))
        self.decoder.add(Conv1D(10, 1, activation="linear"))
        self.decoder.add(Conv1D(1000, 1, activation="linear"))
        self.decoder.add(Conv1D(2000, 1, activation="linear"))
        self.decoder.add(Flatten())
        self.decoder.add(Dense(self.window_size, activation="linear"))
        self.decoder.add(Dense(self.window_size, activation="linear"))

        encoder_input = Input(shape=(self.window_size,))

        encoder = self.encoder(encoder_input)

        decoder = self.decoder(encoder)

        # opt = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"], beta_1=config["beta_1"], beta_2=config["beta_2"])
        self.model = Model(inputs=encoder_input, outputs=decoder)
        self.model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

    def summary(self):
        print(self.model.summary(expand_nested=True))
