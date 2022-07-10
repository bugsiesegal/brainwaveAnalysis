import os

from sklearn.preprocessing import RobustScaler

import datatypes

import ai_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import wandb
from keras import backend as be
from wandb.keras import WandbCallback

hyperparameter_defaults = dict(
    initial_learning_rate=1e-4,
    decay_steps=20000,
    decay_rate=0.9
    )


wandb.init(project="Fiber-Photometry-Autoencoder", entity="bugsiesegal", config=hyperparameter_defaults)

config = wandb.config

autoencoder = ai_model.CNNAutoEncoderModel(100)

autoencoder.make_model(config)
autoencoder.summary()

data = datatypes.FiberPhotometryWindowData.read("C:\\Users\\bugsi\\PycharmProjects\\brainwaveAnalysis\\Data"
                                                         "\\fpw\\0.fpw")

training_data = tf.data.Dataset.from_tensor_slices((data.data, data.data))

training_data = training_data.batch(32)

trained_model_artifact = wandb.Artifact("CNNautoencoderV1", type="model")

# scaler = RobustScaler()
#
# scaler.fit(training_data.data)
#
# data = scaler.transform(training_data.data)

# X_train, X_test, y_train, y_test = train_test_split(data, data, test_size=0.33,
#                                                     random_state=42)

autoencoder.model.fit(training_data, epochs=1,
                      callbacks=[WandbCallback()])

autoencoder.model.evaluate(training_data)

autoencoder.model.save("C:/Users/bugsi/PycharmProjects/brainwaveAnalysis/Data/models/autoencoder.h5")

trained_model_artifact.add_dir("C:/Users/bugsi/PycharmProjects/brainwaveAnalysis/Data/models/")
wandb.run.log_artifact(trained_model_artifact)

fig, axs = plt.subplots(3)

axs[0].plot(data.data[0])
axs[1].plot(autoencoder.model.predict(data.data[0].reshape((1, -1))).reshape((-1,)))
axs[2].bar([i for i in range(10)], autoencoder.encoder.predict(data.data[0].reshape((1, -1))).reshape((-1,)))

wandb.log({"output figure": plt.gcf()})

wandb.finish()
