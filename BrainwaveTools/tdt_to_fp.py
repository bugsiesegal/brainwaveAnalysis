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
    parser.add_argument('block', nargs=1, type=str)
    parser.add_argument('output', nargs=1, type=str)
    return parser


def tdt_to_fiber_photometry_data(path: str):
    data = tdt.read_block(path)
    return FiberPhotometryData(data=data.streams.LMag.data, fs=data.streams.LMag.fs)


def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()
    print(args)
    tdt_to_fiber_photometry_data(args.block[0]).write(args.output[0])


if __name__ == "__main__":
    main()
