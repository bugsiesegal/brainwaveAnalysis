import tensorflow as tf
from keras import Sequential, Model, Input
from keras.layers import Dense, Flatten, Dropout, Bidirectional
from keras.layers import LayerNormalization
from keras.layers.reshaping import Reshape, RepeatVector
from keras.layers.rnn import TimeDistributed, LSTM


class RNNAutoEncoderModel:
    model: Model
    encoder: Model
    decoder: Model

    def __init__(self, window_size, encoded_size):
        self.window_size = window_size
        self.encoded_size = encoded_size

    def make_model(self, config):
        # Extract configuration values
        kernel_initializer = config["kernel_initializer"]
        learning_rate = config["learning_rate"]
        loss = config["loss"]
        dropout = config["dropout"]

        # Encoder
        enc_input = Input((self.window_size,))
        # x = preprocessing_layer(enc_input)
        x = Reshape((-1, 1))(enc_input)

        # Bidirectional LSTM layers to better capture temporal dependencies
        x, _, _, _, _ = Bidirectional(LSTM(self.encoded_size, return_sequences=True, return_state=True,
                                           kernel_initializer=kernel_initializer))(x)
        x = Dropout(dropout)(x)
        x, _, _, _, _ = Bidirectional(LSTM(self.encoded_size, return_sequences=True, return_state=True,
                                           kernel_initializer=kernel_initializer))(x)
        x = Dropout(dropout)(x)
        x, _, _, _, _ = Bidirectional(LSTM(self.encoded_size, return_sequences=True, return_state=True,
                                           kernel_initializer=kernel_initializer))(x)
        x = Dropout(dropout)(x)
        x, _, _, _, _ = Bidirectional(LSTM(self.encoded_size, return_sequences=True, return_state=True,
                                           kernel_initializer=kernel_initializer))(x)
        x = Dropout(dropout)(x)
        x, _, _, _, _ = Bidirectional(LSTM(self.encoded_size, return_sequences=True, return_state=True,
                                           kernel_initializer=kernel_initializer))(x)

        x = Dropout(dropout)(x)
        x = LSTM(self.encoded_size,
                 kernel_initializer=kernel_initializer)(x)
        enc_out = Dense(self.encoded_size, activation="sigmoid",
                        kernel_initializer=kernel_initializer)(x)

        # Decoder
        dec_input = Dense(self.encoded_size, activation="sigmoid",
                          kernel_initializer=kernel_initializer)(enc_out)
        x = RepeatVector(self.window_size)(dec_input)

        x, _, _, _, _ = Bidirectional(LSTM(self.encoded_size, return_sequences=True, return_state=True,
                                           kernel_initializer=kernel_initializer))(x)
        x = Dropout(dropout)(x)
        x, _, _, _, _ = Bidirectional(LSTM(self.encoded_size, return_sequences=True, return_state=True,
                                           kernel_initializer=kernel_initializer))(x)
        x = Dropout(dropout)(x)
        x, _, _, _, _ = Bidirectional(LSTM(self.encoded_size, return_sequences=True, return_state=True,
                                           kernel_initializer=kernel_initializer))(x)
        x = Dropout(dropout)(x)
        x, _, _, _, _ = Bidirectional(LSTM(self.encoded_size, return_sequences=True, return_state=True,
                                           kernel_initializer=kernel_initializer))(x)
        x = Dropout(dropout)(x)
        x, _, _, _, _ = Bidirectional(LSTM(self.encoded_size, return_sequences=True, return_state=True,
                                           kernel_initializer=kernel_initializer))(x)
        x = Dropout(dropout)(x)
        x = LSTM(self.encoded_size, return_sequences=True,
                 kernel_initializer=kernel_initializer)(x)
        x = TimeDistributed(Dense(1, activation="sigmoid",
                                  kernel_initializer=kernel_initializer))(x)
        dec_out = Reshape((-1,))(x)

        # Model compilation
        self.model = Model(inputs=enc_input, outputs=dec_out)
        self.encoder = Model(inputs=enc_input, outputs=enc_out)
        self.decoder = Model(inputs=dec_input, outputs=dec_out)
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=opt, loss=loss)

    def summary(self):
        print(self.model.summary())


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
        self.model.compile(optimizer=opt, loss=config["loss"], metrics=['accuracy'])

    def summary(self):
        print(self.model.summary(expand_nested=True))
