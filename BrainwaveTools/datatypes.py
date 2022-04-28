from dataclasses import dataclass
import numpy as np


@dataclass
class FiberPhotometryData:
    """
    fs: Data Sampling Rate.
    frequency: Recording frequency.
    time: Data recording time.


    """
    fs: np.float
    data: np.ndarray

    @property
    def frequency(self) -> np.float:
        return 1/self.fs

    @property
    def time(self) -> np.float:
        return self.data.shape[0] * self.frequency

    @property
    def shape(self) -> tuple:
        return self.data.shape
