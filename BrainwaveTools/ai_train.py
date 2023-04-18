import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import os
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

import ai_model
import datatypes


hyperparameter_defaults = dict(
    epochs=100,
    window_size=10000,
    compression_size=10,
    learning_rate=1e-4,
    dropout=0.4,
    loss="mse",
    kernel_initializer="he_normal"
)

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

tf.compat.v1.Session(config=config)

run = wandb.init(project="brainwaveAnalysis-BrainwaveTools-RNN", entity="bugsiesegal", config=hyperparameter_defaults)

config = wandb.config

print(config)

autoencoder = ai_model.RNNAutoEncoderModel(config["window_size"], config["compression_size"])

autoencoder.make_model(config)
autoencoder.summary()


# autoencoder.model.load_weights("/home/bugsie/PycharmProjects/brainwaveAnalysis/Models/model.h5")

def read_and_preprocess_data(filepath):
    fp_data = datatypes.FiberPhotometryData.read(filepath)
    data = fp_data.data[0]
    return data, fp_data


def create_timeseries_dataset(data, window_size):
    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data[:-window_size], data[:-window_size], window_size, sequence_stride=100, shuffle=True,
        batch_size=2 * 8
    )
    return dataset


def apply_sliding_window(fp_data, window_size):
    sliding_window_data = fp_data.sliding_window(window_size).data
    return sliding_window_data

def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data.reshape(-1, 1))
    return normalized_data

X_train_file = "/home/bugsie/PycharmProjects/brainwaveAnalysis/Test/fp/1.fp"
y_test_file = "/home/bugsie/PycharmProjects/brainwaveAnalysis/Test/fp/3.fp"



# Read and preprocess the data
X_train, _ = read_and_preprocess_data(X_train_file)
y_test, y_test_fp = read_and_preprocess_data(y_test_file)
z_data = apply_sliding_window(_, config["window_size"])

# Normalize the data
X_train = normalize_data(X_train)
y_test = normalize_data(y_test)
z_data = normalize_data(z_data.reshape(-1, 1)).reshape(-1, config["window_size"])

print(pd.DataFrame(X_train).describe())
print(pd.DataFrame(z_data).describe())

X_train_dataset = create_timeseries_dataset(X_train[:-20000], config["window_size"])
X_val_dataset = create_timeseries_dataset(X_train[-20000:], config["window_size"])
y_test_dataset = create_timeseries_dataset(X_train[:200000], config["window_size"])

print(pd.DataFrame(z_data).shape)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="/home/bugsie/PycharmProjects/brainwaveAnalysis/Models/model.h5",
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)

autoencoder.model.fit(X_train_dataset, epochs=config["epochs"], validation_data=X_val_dataset,
                      callbacks=[WandbCallback(save_model=False), model_checkpoint_callback])

autoencoder.encoder.save("/home/bugsie/PycharmProjects/brainwaveAnalysis/Models/enc_model.h5")

wandb.log({"test_loss": np.sum(autoencoder.model.evaluate(y_test_dataset))})

pred = autoencoder.encoder.predict(y_test_dataset, verbose=0)

for i in range(10):
    plt.plot(pred.T[i])

# wandb.log({"Pred": px.line(pred.T[0])})

wandb.log({"Actual": px.line(pd.DataFrame(z_data[30].reshape((-1, 1))))})
wandb.log({"Recreation": px.line(pd.DataFrame(autoencoder.model.predict(z_data[30].reshape((1, -1)), verbose=0)).T)})
wandb.log(
    {"Encoded Values": plt.gcf()})
wandb.finish()
