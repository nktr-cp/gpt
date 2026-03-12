"""Project entrypoint for manual experiments."""

from argparse import ArgumentParser
from collections.abc import Sequence


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="gpt", description="GPT implementation workspace.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    build_parser().parse_args(argv)
