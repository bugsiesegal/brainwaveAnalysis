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
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

hyperparameter_defaults = dict(
    epochs=10000,
    window_size=10000,
    compression_size=10,
    learning_rate=6e-3,
    initial_accumulator_value=0.95,
    dropout=0.2
)

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

tf.compat.v1.Session(config=config)

run = wandb.init(project="brainwaveAnalysis-BrainwaveTools", entity="bugsiesegal", config=hyperparameter_defaults)

config = wandb.config

print(config)

autoencoder = ai_model.AutoEncoderModel(config["window_size"], config["compression_size"], config["dropout"])

autoencoder.make_model(config)
autoencoder.summary()

autoencoder.model.load_weights("/home/bugsie/PycharmProjects/brainwaveAnalysis/Models/model.h5")

data = datatypes.FiberPhotometryWindowData.read("/home/bugsie/PycharmProjects/brainwaveAnalysis/Test/fpw/10000/1.fpw").data

data = data[np.where(np.any(data > 0.2, axis=1))]

scaler = MinMaxScaler()

data = scaler.fit_transform(data)

desc = pd.DataFrame(data.T).describe()

figs, axs = plt.subplots(1, 3)

axs[0].plot(pd.DataFrame(data[desc.idxmax(axis=1)][:6]).T)
axs[1].plot(pd.DataFrame(autoencoder.model.predict(data[desc.idxmax(axis=1)][:6], verbose=0)).T)
print(autoencoder.encoder.predict(data[desc.idxmax(axis=1)][:6], verbose=0).shape)
axs[2].plot(autoencoder.encoder.predict(data[desc.idxmax(axis=1)][:6], verbose=0))

plt.show()
