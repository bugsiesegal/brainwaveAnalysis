import os.path
from glob import glob

import tdt
import argparse
from datatypes import FiberPhotometryData


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Converts TDT block to FiberPhotometryData File"
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument(
        "-m", "--multiple", action="store_true"
    )
    parser.add_argument('block', nargs=1, type=str)
    parser.add_argument('output', nargs=1, type=str)
    return parser


def tdt_to_fiber_photometry_data(path: str, multiple=False):
    if not multiple:
        data = [tdt.read_block(path)]
    else:
        data = []
        for path in glob(os.path.join(path, "*")):
            data.append(tdt.read_block(path))

    return [FiberPhotometryData(data=d.streams.LMag.data, fs=d.streams.LMag.fs) for d in data]


def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()
    for i, data in enumerate(tdt_to_fiber_photometry_data(args.block[0], multiple=args.multiple)):
        data.write(os.path.join(args.output[0], str(i) + ".fp"))


if __name__ == "__main__":
    main()
