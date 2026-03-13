"""Project entrypoint for manual experiments."""

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from .data.download import dataset_path, download_names_dataset
from .dataset import build_token_stream, build_tokenizer, load_documents, sample_next_token_batch


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

    batch_parser = subparsers.add_parser(
        "inspect-batch",
        help="Sample a next-token prediction batch from the flattened token stream.",
    )
    batch_parser.add_argument(
        "--dataset",
        type=Path,
        default=dataset_path(),
        help="Path to the plain text dataset.",
    )
    batch_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of sequences to sample.",
    )
    batch_parser.add_argument(
        "--block-size",
        type=int,
        default=8,
        help="Context length for each training example.",
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
        return

    if args.command == "inspect-batch":
        documents = load_documents(args.dataset)
        tokenizer = build_tokenizer(documents)
        token_stream = build_token_stream(documents, tokenizer)
        x, y = sample_next_token_batch(
            token_stream,
            batch_size=args.batch_size,
            block_size=args.block_size,
        )
        print(f"token_stream_shape={tuple(token_stream.shape)}")
        print(f"x_shape={tuple(x.shape)}")
        print(f"y_shape={tuple(y.shape)}")
        print(f"x[0]={x[0].tolist()}")
        print(f"y[0]={y[0].tolist()}")
