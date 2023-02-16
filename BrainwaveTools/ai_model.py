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
            sequential.add(Dense(int(100), activation="sigmoid"))

        x = sequential(x)
        x = Flatten()(x)
        enc_out = Dense(self.encoded_size, activation="sigmoid")(x)

        dec_in = Dense(self.window_size, activation="sigmoid")(enc_out)
        x = Reshape((100, -1))(dec_in)
        sequential = Sequential()

        for i in range(self.num_hidden_layers):
            sequential.add(Dense(int(100), activation="sigmoid"))

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