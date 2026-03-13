"""Project entrypoint for manual experiments."""

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from .data.download import dataset_path, download_names_dataset
from .dataset import build_tokenizer, load_documents


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

    inspect_parser = subparsers.add_parser(
        "inspect-tokenizer",
        help="Build a character tokenizer and show a round-trip example.",
    )
    inspect_parser.add_argument(
        "--dataset",
        type=Path,
        default=dataset_path(),
        help="Path to the plain text dataset.",
    )
    inspect_parser.add_argument(
        "--text",
        default="anna",
        help="Text used for the encode/decode sanity check.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "download-names":
        print(download_names_dataset(destination=args.output))
        return

    if args.command == "inspect-tokenizer":
        documents = load_documents(args.dataset)
        tokenizer = build_tokenizer(documents)
        encoded = tokenizer.encode(args.text, add_bos=True, add_eos=True)
        decoded = tokenizer.decode(encoded)
        print(f"vocab_size={tokenizer.vocab_size}")
        print(f"encoded={encoded}")
        print(f"decoded={decoded}")
