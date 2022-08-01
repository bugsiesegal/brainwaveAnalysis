import math

import tensorflow as tf
from tensorflow import keras
from keras.layers.reshaping import Reshape
from keras.layers.rnn import TimeDistributed
from keras.layers.convolutional import Convolution1DTranspose, Conv1D
from keras import Sequential, Model, Input
from keras.optimizers import Adam
from keras.layers import Dense, MaxPool1D, Flatten, Attention, BatchNormalization
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow_compression as tfc
import wandb
from wandb.keras import WandbCallback

from BrainwaveTools import datatypes
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(tf.config.list_physical_devices())

wandb.init(project="Fiber-Photometry-Autoencoder", entity="bugsiesegal")

def make_encoder(latent_dims):
    return Sequential([
        Reshape((1, 100)),
        Conv1D(2000, 1, strides=2, use_bias=True, padding="same", activation="leaky_relu"),
        Conv1D(5000, 1, strides=2, use_bias=True, padding="same", activation="leaky_relu"),
        Flatten(),
        Dense(100, use_bias=True, activation="leaky_relu"),
        Dense(latent_dims, use_bias=True, activation=None)
    ], name="encoder")


def make_decoder():
    return Sequential([
        Dense(100, use_bias=True, activation="leaky_relu"),
        Dense(1000, use_bias=True, activation="leaky_relu"),
        Reshape((1, 1000)),
        Convolution1DTranspose(2000, 1, strides=2, padding="same", activation="leaky_relu"),
        Convolution1DTranspose(25, 1, strides=2, padding="same", activation="leaky_relu"),
        Flatten()
    ])


class FiberPhotometryAutoencoder(Model):

    def __init__(self, latent_dims):
        super().__init__()
        self.encoder = make_encoder(latent_dims)
        self.decoder = make_decoder()
        self.prior_log_scales = tf.Variable(tf.zeros((latent_dims,)))

    @property
    def prior(self):
        return tfc.NoisyLogistic(loc=0, scale=tf.exp(self.prior_log_scales))

    def call(self, x, training):
        x = tf.cast(x, self.compute_dtype)
        x = tf.reshape(x, (-1, 1, 100))

        y = self.encoder(x)
        entropy_model = tfc.ContinuousBatchedEntropyModel(
            self.prior, coding_rank=1, compression=False
        )
        y_tilde, rate = entropy_model(y, training=training)
        x_tilde = self.decoder(y_tilde)

        rate = tf.reduce_mean(rate)

        distortion = tf.reduce_mean(abs(x - x_tilde))

        return dict(rate=rate, distortion=distortion)


def pass_through_loss(_, x):
    return x

training_data = datatypes.FiberPhotometryWindowData.read("/mnt/c/Users/bugsi/PycharmProjects/brainwaveAnalysis/Test"
                                                         "/fpw/0.fpw")

X_train, X_test, y_train, y_test = train_test_split(training_data.data[:200], training_data.data[:200], test_size=0.33,
                                                    random_state=42)

training_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

def make_compression_trainer(lmbda, latent_dims=10):
    trainer = FiberPhotometryAutoencoder(latent_dims)
    trainer.compile(
        optimizer=Adam(learning_rate=1e-3),

        loss=dict(rate=pass_through_loss, distortion=pass_through_loss),
        metrics=dict(rate=pass_through_loss, distortion=pass_through_loss),
        loss_weights=dict(rate=1., distortion=lmbda),
    )
    return trainer


def trainer_model(lmbda):
    trainer = make_compression_trainer(lmbda)
    trainer.fit(
        training_dataset.batch(128).prefetch(8),
        epochs=15,
        validation_data=test_dataset.batch(128).cache(),
        validation_freq=1,
        verbose=1,
        callbacks=[]
    )

    return trainer

trainer = trainer_model(lmbda=2000)

class Encoder(Model):
    def __init__(self, analysis_transform, entropy_model):
        super().__init__()
        self.entropy_model = entropy_model
        self.analysis_transform = analysis_transform

    def call(self, x):
        x = tf.cast(x, self.compute_dtype)
        y = self.analysis_transform(x)

        _, bits = self.entropy_model(y, training=False)

        return self.entropy_model.compress(y), bits

class Decoder(Model):
    def __init__(self, synthesis_transform, entropy_model):
        super().__init__()
        self.entropy_model = entropy_model
        self.synthesis_transform = synthesis_transform

    def call(self, string):
        y_hat = self.entropy_model.decompress(string, ())
        x_hat = self.synthesis_transform(y_hat)

        return x_hat

def make_codec(trainer, **kwargs):
    entropy_model = tfc.ContinuousBatchedEntropyModel(
        trainer.prior, coding_rank=1, compression=True, **kwargs
    )
    compressor = Encoder(trainer.encoder, entropy_model)
    decompressor = Decoder(trainer.decoder, entropy_model)

    return compressor, decompressor

compressor, decompressor = make_codec(trainer)

(originals, _), = test_dataset.batch(16).skip(3).take(1)

strings, entropies = compressor(originals)

print(f"String representation of first digit in hexadecimal: 0x{strings[0].numpy().hex()}")
print(f"Number of bits actually needed to represent it: {entropies[0]:0.2f}")

reconstructions = decompressor(strings)

fig, axs = plt.subplots(3)

axs[0].plot(originals[0])
axs[1].plot(reconstructions[0])

wandb.log({"plot": fig})

