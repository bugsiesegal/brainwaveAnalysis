import os.path
from glob import glob

import numpy as np
import tdt
import argparse
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from datatypes import FiberPhotometryData, FiberPhotometryWindowData
from numpy.lib.stride_tricks import sliding_window_view


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
    parser.add_argument('output', nargs=1, type=str)
    parser.add_argument('-l', '--list', nargs='+', help='<Required> Set flag', required=True)

    return parser


def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()
    print(args)
    enc_model = keras.models.load_model("/home/bugsie/PycharmProjects/brainwaveAnalysis/Models/enc_model.h5")
    model = keras.models.load_model("/home/bugsie/PycharmProjects/brainwaveAnalysis/Models/model.h5")

    for timestamp in args.list:
        window_data = FiberPhotometryData.read(args.input[0])
        data = window_data.data

        scaler = MinMaxScaler()

        time_index = int((int(timestamp) / 1000) / window_data.frequency)

        print(data[0, time_index - 5002:time_index + 5002].shape)

        data = sliding_window_view(data[0, time_index - 5002:time_index + 5002], 10000)

        data = scaler.fit_transform(data)

        print(time_index)
        print(data.shape)

        predicted_enc_data = enc_model.predict(data)
        predicted_data = model.predict(data)

        figs, axs = plt.subplots(3)
        print((predicted_enc_data[0] - predicted_enc_data[-1]).shape)
        axs[0].bar([i for i in range(10)], predicted_enc_data[0] - predicted_enc_data[-1])
        axs[1].plot(data[0])
        axs[2].plot(predicted_data[0])
        print(args.output[0] + str(timestamp) + ".png")
        plt.savefig(args.output[0] + str(timestamp) + ".png")


if __name__ == "__main__":
    main()
