import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import quantile_transform

from BrainwaveTools import datatypes

data = datatypes.FiberPhotometryWindowData.read("/root/PycharmProjects/brainwaveAnalysis/Test/fpw/0.fpw").data

idx = np.random.randint(data.shape[0], size=2000)

data = data[idx, :]

print(data.shape)

print(np.where(np.any(data > 0.2, axis=1)))

data = data[np.where(np.any(data > 0.2, axis=1))]

print(data.shape)

q_data = quantile_transform(data, n_quantiles=10, random_state=0)

print(np.mgrid[0:data.shape[0], 0:data.shape[1]].shape)

_, (train_ax, q_pca_ax) = plt.subplots(ncols=2, figsize=(8, 4))
train_ax.scatter(np.mgrid[0:data.shape[0], 0:data.shape[1]][1].T, data, alpha=0.02)

print(q_data.shape)
print(np.mgrid[0:q_data.shape[0], 0:q_data.shape[1]].shape)

q_pca_ax.scatter(np.mgrid[0:q_data.shape[0], 0:q_data.shape[1]][1].T, q_data, alpha=0.01)

plt.show()
