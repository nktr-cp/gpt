"""Project entrypoint for manual experiments."""

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from .data.download import dataset_path, download_names_dataset


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="gpt", description="GPT implementation workspace.")
    subparsers = parser.add_subparsers(dest="command")

    download_parser = subparsers.add_parser(
        "download-names",
        help="Download the small names dataset into data/raw.",
    )
    download_parser.add_argument(
        "--output",
        type=Path,
        default=dataset_path(),
        help="Where to store the downloaded dataset.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "download-names":
        print(download_names_dataset(destination=args.output))
