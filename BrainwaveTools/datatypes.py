from __future__ import annotations

import dill
from dataclasses import dataclass

import numpy as np

from numpy.lib.stride_tricks import sliding_window_view


class FileMixin:
    def write(self, path):
        with open(path, 'wb') as f:
            dill.dump(self, f, protocol=2)

    @classmethod
    def read(cls, path) -> FileMixin:
        with open(path, 'rb') as f:
            return dill.load(f)


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

    def sliding_window(self, window_shape: int) -> np.ndarray:
        return sliding_window_view(self.data.flatten(), window_shape)
