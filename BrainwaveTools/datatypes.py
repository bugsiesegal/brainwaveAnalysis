import dill
from dataclasses import dataclass

import numpy as np

from numpy.lib.stride_tricks import sliding_window_view


class FileMixin:
    def write(self, path):
        with open(path, 'wb') as f:
            dill.dump(self, f, protocol=2)

    @classmethod
    def read(cls, path) -> "FileMixin":
        with open(path, 'rb') as f:
            return dill.load(f)


class FiberPhotometryWindowData(FileMixin):
    """
    fs: Data Sampling Rate.
    frequency: Recording frequency.
    time: Data recording time.
    shape: Data shape.
    """

    fs: np.float
    data: np.ndarray
    time_array: np.ndarray

    def __init__(self, fs, data, time):
        self.fs = fs
        self.data = data
        self.time_array = time

    @property
    def frequency(self) -> np.float:
        return 1 / self.fs

    @property
    def time(self) -> np.float:
        return self.data.shape[0] * self.frequency

    @property
    def shape(self) -> tuple:
        return self.data.shape


@dataclass
class FiberPhotometryData(FileMixin):
    """
    fs: Data Sampling Rate.
    frequency: Recording frequency.
    time: Data recording time.
    shape: Data shape.
    """

    fs: np.float
    data: np.ndarray

    @property
    def frequency(self) -> np.float:
        return 1 / self.fs

    @property
    def time(self) -> np.float:
        return self.data.shape[0] * self.frequency

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def sliding_window(self, window_shape: int) -> FiberPhotometryWindowData:
        w_data = self.data.flatten()[:self.data.flatten().size - (self.data.flatten().size % 10000)].reshape((-1, window_shape))
        t_data = np.arange(0, self.data.flatten().size - (self.data.flatten().size % 10000), dtype=int).reshape((-1, window_shape))
        return FiberPhotometryWindowData(fs=self.fs,
                                         data=w_data,
                                         time=t_data
                                         )
        # return FiberPhotometryWindowData(self.fs, sliding_window_view(self.data.flatten(), window_shape))
