import os.path
from glob import glob

import numpy as np
import tdt
import argparse
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from datatypes import FiberPhotometryData, FiberPhotometryWindowData


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Converts FiberPhotometryData File to FiberPhotometryWindowData"
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument('input', nargs=1, type=str)
    parser.add_argument('time_stamp', nargs=1, type=float)
    return parser


def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()
    window_data = FiberPhotometryWindowData.read(args.input[0])
    data = window_data.data

    scaler = MinMaxScaler()

    data = scaler.fit_transform(data)

    time = window_data.time_array
    time_index = np.where(np.any(time == int(args.time_stamp[0]), axis=1))[0]
    print(time_index)
    print(data[time_index][0])
    model = keras.models.load_model("/home/bugsie/PycharmProjects/brainwaveAnalysis/Models/model.h5")
    plt.plot(data[time_index][0])
    plt.plot(model.predict(data[time_index])[0])
    plt.show()




if __name__ == "__main__":
    main()
