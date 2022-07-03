import math

import tensorflow as tf
from tensorflow import keras
from keras.layers.reshaping import Reshape
from keras.layers.rnn import TimeDistributed
from keras.layers.convolutional import Convolution1DTranspose, Conv1D
from keras import Sequential, Model, Input
from keras.layers import Dense


class AutoEncoderModel:
    model: Model
    encoder: Model
    decoder: Model

    def __init__(self, window_size, encoded_size, num_hidden_layers=2):
        self.window_size = window_size
        self.encoded_size = encoded_size
        self.num_hidden_layers = num_hidden_layers

    def make_model(self):
        enc_in = Input((self.window_size,))
        x = Sequential([
            Dense(self.window_size * 4) for i in range(self.num_hidden_layers)
        ])(enc_in)
        enc_out = Dense(self.encoded_size)(x)
        dec_in = Dense(self.encoded_size)(enc_out)
        x = Sequential([
            Dense(self.window_size * 4) for i in range(self.num_hidden_layers)
        ])(dec_in)
        dec_out = Dense(self.window_size)(x)

        self.encoder = Model(inputs=enc_in, outputs=enc_out)
        self.decoder = Model(inputs=dec_in, outputs=dec_out)
        self.model = Model(inputs=enc_in, outputs=dec_out)
        opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
        self.model.compile(optimizer=opt, loss="mse", metrics=[tf.keras.metrics.Accuracy()])

    def summary(self):
        print(self.model.summary())
