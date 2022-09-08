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
    epochs=1000,
    window_size=10000,
    compression_size=100,
    learning_rate=4e-2,
    initial_accumulator_value=0.95
)

tf.compat.v1.disable_eager_execution()
run = wandb.init(project="brainwaveAnalysis-BrainwaveTools", entity="bugsiesegal", config=hyperparameter_defaults)

config = wandb.config

print(config)

autoencoder = ai_model.CNNAutoEncoderModel(config["window_size"], config["compression_size"])

autoencoder.make_model(config)
autoencoder.summary()

# autoencoder.model.load_weights("/home/bugsie/PycharmProjects/brainwaveAnalysis/Models/model.h5")

data = datatypes.FiberPhotometryWindowData.read("/home/bugsie/PycharmProjects/brainwaveAnalysis/Test/fpw/10000/0.fpw").data

idx = np.random.randint(data.shape[0], size=2 ** 10)

data = data[idx, :]

data = data[np.where(np.any(data > 0.2, axis=1))]

desc = pd.DataFrame(data.T).describe()

print(desc)

X_train, X_test, y_train, y_test = train_test_split(data, data,
                                                    test_size=0.33,
                                                    random_state=42)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="/home/bugsie/PycharmProjects/brainwaveAnalysis/Models/model.h5",
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

autoencoder.model.fit(X_train, y_train, epochs=config["epochs"], validation_split=0.1, batch_size=2 ** 14,
                      callbacks=[WandbCallback(save_model=False)])

# autoencoder.model.evaluate(X_test, y_test)

wandb.log({"Actual": px.line(pd.DataFrame(data[desc.idxmax(axis=1)][:4]).T)})
wandb.log({"Recreation": px.line(pd.DataFrame(autoencoder.model.predict(data[desc.idxmax(axis=1)][:4], verbose=0)).T)})
wandb.log(
    {"Encoded Values": px.bar(pd.DataFrame(autoencoder.encoder.predict(data[desc.idxmax(axis=1)][:4], verbose=0).T), barmode='group')})
wandb.finish()
