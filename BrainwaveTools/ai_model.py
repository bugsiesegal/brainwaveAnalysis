import math

import tensorflow as tf
from tensorflow import keras
from keras.layers.reshaping import Reshape
from keras.layers.rnn import TimeDistributed
from keras.layers.convolutional import Convolution1DTranspose, Conv1D
from keras import Sequential, Model, Input
from keras.layers import Dense


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
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config["initial_learning_rate"],
            decay_steps=config["decay_step"],
            decay_rate=config["decay_rate"])
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        lr_metric = get_lr_metric(opt)
        self.model.compile(optimizer=opt, loss="mse", metrics=['binary_accuracy', lr_metric])

    def summary(self):
        print(self.model.summary())


class CNNAutoEncoderModel:
    model: Model
    encoder: Model
    decoder: Model

    def __init__(self, window_size):
        self.window_size = window_size

    def make_model(self, config):
        enc_in = Input((self.window_size,))
        x = Reshape((1, self.window_size))(enc_in)
        x1 = Sequential([
                            Convolution1DTranspose(10, 10) for i in range(6)
                        ] + [
                            Conv1D(10, 10) for i in range(6)
                        ])(x)
        enc_out = Dense(10, activation="sigmoid")(x1)
        dec_in = Dense(10)(enc_out)
        x2 = Sequential([Convolution1DTranspose(10, 10) for i in range(6)]
                        + [Conv1D(10, 10) for i in range(5)])(dec_in)
        # x2 = Dense(100)(x2)
        x3 = Reshape((self.window_size,))(x2)
        dec_out = Dense(self.window_size, activation="sigmoid")(x3)

        self.encoder = Model(inputs=enc_in, outputs=enc_out)
        self.decoder = Model(inputs=dec_in, outputs=dec_out)
        self.model = Model(inputs=enc_in, outputs=dec_out)
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config["initial_learning_rate"],
            decay_steps=config["decay_steps"],
            decay_rate=config["decay_rate"])
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        lr_metric = get_lr_metric(opt)
        self.model.compile(optimizer=opt, loss="mse", metrics=['binary_accuracy', lr_metric])

    def summary(self):
        print(self.model.summary(expand_nested=True))
