import argparse

import numpy as np

from datatypes import FiberPhotometryData, FiberPhotometryWindowData
import matplotlib.pyplot as plt


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Converts TDT block to FiberPhotometryData File"
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument('file', nargs=1, type=str)
    return parser


def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()
    fp = FiberPhotometryData.read(args.file[0])

    ticks = np.arange(fp.shape[1]) * fp.frequency

    plt.plot(ticks, fp.data[0])

    plt.xlabel("Time (s)")

    plt.show()


if __name__ == "__main__":
    main()
