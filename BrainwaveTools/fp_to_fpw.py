import os.path
from glob import glob

import tdt
import argparse
from datatypes import FiberPhotometryData


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Converts FiberPhotometryData File to FiberPhotometryWindowData"
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument(
        "-m", "--multiple", action="store_true"
    )
    parser.add_argument('input', nargs=1, type=str)
    parser.add_argument('output', nargs=1, type=str)
    parser.add_argument('window_size', nargs=1, type=int)
    return parser


def fiber_photometry_data_to_fiber_photometry_window_data(path: str, window_size: int, multiple=False):
    if not multiple:
        data = [FiberPhotometryData.read(path)]
    else:
        data = []
        for path in glob(os.path.join(path, "*")):
            data.append(FiberPhotometryData.read(path))

    return [d.sliding_window(window_size) for d in data]


def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()
    for i, data in enumerate(
            fiber_photometry_data_to_fiber_photometry_window_data(
                args.input[0],
                args.window_size[0],
                multiple=args.multiple)):
        data.write(os.path.join(args.output[0], str(i) + ".fpw"))


if __name__ == "__main__":
    main()
