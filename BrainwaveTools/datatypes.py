from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np


class FileMixin:
    def write(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=2)

    @classmethod
    def read(cls, path) -> FileMixin:
        with open(path, 'rb') as f:
            return pickle.load(f)


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
