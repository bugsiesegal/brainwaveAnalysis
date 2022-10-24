import math
import os

import keras.models
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import RobustScaler, MinMaxScaler, quantile_transform, QuantileTransformer

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
    epochs=10000,
    window_size=10000,
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

X_train = datatypes.FiberPhotometryWindowData.read("/home/bugsie/PycharmProjects/brainwaveAnalysis/Test/fpw/10000/2.fpw").data

y_test = datatypes.FiberPhotometryWindowData.read("/home/bugsie/PycharmProjects/brainwaveAnalysis/Test/fpw/10000/1.fpw").data

X_train = X_train[np.where(np.all(X_train > 0.2, axis=1))]

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

desc = pd.DataFrame(X_train.T).describe()

print(desc)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="/home/bugsie/PycharmProjects/brainwaveAnalysis/Models/model.h5",
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

autoencoder.model.fit(X_train, X_train, epochs=config["epochs"], validation_split=0.1, batch_size=2 ** 8,
                      callbacks=[WandbCallback(save_model=False), model_checkpoint_callback])

wandb.log({"test_loss": autoencoder.model.evaluate(y_test, y_test)[0]})

wandb.log({"Actual": px.line(pd.DataFrame(y_test[:3]).T)})
wandb.log({"Recreation": px.line(pd.DataFrame(autoencoder.model.predict(y_test[:3], verbose=0)).T)})
wandb.log(
    {"Encoded Values": px.bar(pd.DataFrame(autoencoder.encoder.predict(y_test[:3], verbose=0)), barmode='group')})
wandb.finish()
