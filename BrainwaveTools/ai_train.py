import math
import os

import keras.models
from sklearn.preprocessing import RobustScaler, MinMaxScaler

import datatypes

import ai_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import wandb
from keras import backend as be
from wandb.keras import WandbCallback

hyperparameter_defaults = dict(
    learning_rate=0.017089103647893028,
    initial_accumulator_value=0.1,
    epochs=1000,
    window_size=100,
    compression_size=10,
    hidden_layer_stacks=4
)

# tf.compat.v1.disable_eager_execution()
run = wandb.init(project="brainwaveAnalysis-BrainwaveTools", entity="bugsiesegal", config=hyperparameter_defaults)

config = wandb.config

print(config)

autoencoder = ai_model.CNNAutoEncoderModel(config["window_size"], config["compression_size"], hidden_layer_stacks=config["hidden_layer_stacks"])

autoencoder.make_model(config)
autoencoder.summary()

# best_model = wandb.restore('model.h5', run_path="bugsiesegal/brainwaveAnalysis-BrainwaveToolsV1.1/2pzv4332")
#
# autoencoder.model.load_weights(best_model.name)

data = datatypes.FiberPhotometryWindowData.read("C:/Users/bugsi/PycharmProjects/brainwaveAnalysis/Test"
                                                "/fpw/0.fpw")

scaler = MinMaxScaler()

data = scaler.fit_transform(data.data)

X_train, X_test, y_train, y_test = train_test_split(data[:20], data[:20],
                                                    test_size=0.33,
                                                    random_state=42)

# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath="C:/Users/bugsi/PycharmProjects/brainwaveAnalysis/Data/models/model.h5",
#     save_weights_only=True,
#     monitor='val_accuracy',
#     mode='max',
#     save_best_only=True)

autoencoder.model.fit(X_train, y_train, epochs=config["epochs"], validation_split=0.1, batch_size=2 ** 10,
                      callbacks=[WandbCallback(save_model=False)])

# autoencoder.model.evaluate(X_test, y_test)

fig, axs = plt.subplots(3)

axs[0].plot(X_train[0])
axs[1].plot(autoencoder.model.predict(X_train[0].reshape((1, -1))).reshape((-1,)))
axs[2].bar([i for i in range(10)], autoencoder.encoder.predict(X_train[0].reshape((1, -1))).reshape((-1,)))

wandb.log({"output figure": plt.gcf()})

# wandb.save('model.h5')

wandb.finish()
