import math
import os

import keras.models
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import RobustScaler, MinMaxScaler, quantile_transform, QuantileTransformer, normalize

import datatypes

import ai_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import wandb
from keras import backend as be
from wandb.keras import WandbCallback
import pandas as pd
import plotly.express as px

hyperparameter_defaults = dict(
    epochs=20,
    window_size=1000,
    compression_size=10,
    learning_rate=1e-4,
    initial_accumulator_value=0.95,
    dropout=0.16,
    num_hidden=6
)

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

tf.compat.v1.Session(config=config)

run = wandb.init(project="brainwaveAnalysis-BrainwaveTools", entity="bugsiesegal", config=hyperparameter_defaults)

config = wandb.config

print(config)

autoencoder = ai_model.AutoEncoderModel(config["window_size"], config["compression_size"], config["dropout"],
                                        num_hidden_layers=config["num_hidden"])

autoencoder.make_model(config)
autoencoder.summary()

# autoencoder.model.load_weights("/home/bugsie/PycharmProjects/brainwaveAnalysis/Models/model.h5")

X_train = datatypes.FiberPhotometryData.read("/home/bugsie/PycharmProjects/brainwaveAnalysis/Test/fp/1.fp").data[0]

y_test = datatypes.FiberPhotometryData.read("/home/bugsie/PycharmProjects/brainwaveAnalysis/Test/fp/1.fp")

z_data = y_test.sliding_window(config["window_size"]).data

y_test = y_test.data[0]

X_train = normalize(X_train.reshape(1,-1)).reshape(-1,)

y_test = normalize(y_test.reshape(1,-1)).reshape(-1,)

print(pd.DataFrame(X_train.T).describe())

print(pd.DataFrame(y_test.T).describe())

X_train_dataset = tf.keras.utils.timeseries_dataset_from_array(X_train[:-10000], X_train[:-10000], config["window_size"])

X_val_dataset = tf.keras.utils.timeseries_dataset_from_array(X_train[-10000:], X_train[-10000:], config["window_size"])

y_test_dataset = tf.keras.utils.timeseries_dataset_from_array(y_test, y_test, config["window_size"])



model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="/home/bugsie/PycharmProjects/brainwaveAnalysis/Models/model.h5",
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

autoencoder.model.fit(X_train_dataset, epochs=config["epochs"], validation_data=X_val_dataset,
                      callbacks=[WandbCallback(save_model=False), model_checkpoint_callback])

autoencoder.encoder.save("/home/bugsie/PycharmProjects/brainwaveAnalysis/Models/enc_model.h5")

wandb.log({"test_loss": autoencoder.model.evaluate(y_test_dataset)[0]})

pred = autoencoder.encoder.predict(y_test_dataset, verbose=0)

print(pred.shape)

print(pred[:10].T)

for i in range(10):
    plt.plot(pred.T[i])

# wandb.log({"Pred": px.line(pred.T[0])})

wandb.log({"Actual": px.line(pd.DataFrame(z_data[30].reshape((1,-1))).T)})
wandb.log({"Recreation": px.line(pd.DataFrame(autoencoder.model.predict(z_data[30].reshape((1,-1)), verbose=0)).T)})
wandb.log(
    {"Encoded Values": plt.gcf()})
wandb.finish()
