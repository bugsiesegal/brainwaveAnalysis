import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import quantile_transform

from BrainwaveTools import datatypes

data = datatypes.FiberPhotometryData.read("/root/PycharmProjects/brainwaveAnalysis/Test/fp/1.fp")

plt.plot(data.data)

plt.show()
