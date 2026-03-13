"""Project entrypoint for manual experiments."""

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from .checkpoint import load_checkpoint, save_checkpoint
from .data.download import dataset_path, download_names_dataset
from .dataset import load_documents
from .generation import generate_text
from .training import TrainingConfig, train_model


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

    train_parser = subparsers.add_parser(
        "train",
        help="Run a minimal GPT training loop on the local names dataset.",
    )
    train_parser.add_argument("--dataset", type=Path, default=dataset_path())
    train_parser.add_argument("--block-size", type=int, default=32)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--n-layer", type=int, default=2)
    train_parser.add_argument("--n-head", type=int, default=4)
    train_parser.add_argument("--n-embd", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--num-steps", type=int, default=200)
    train_parser.add_argument("--log-interval", type=int, default=20)
    train_parser.add_argument("--tokenizer-kind", choices=["char", "bpe"], default="bpe")
    train_parser.add_argument("--bpe-vocab-size", type=int, default=128)
    train_parser.add_argument("--prompt", default="a")
    train_parser.add_argument("--max-new-tokens", type=int, default=24)
    train_parser.add_argument("--temperature", type=float, default=1.0)
    train_parser.add_argument("--checkpoint-out", type=Path)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Load a saved checkpoint and generate text from it.",
    )
    generate_parser.add_argument("checkpoint", type=Path)
    generate_parser.add_argument("--prompt", default="")
    generate_parser.add_argument("--max-new-tokens", type=int, default=24)
    generate_parser.add_argument("--temperature", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "download-names":
        print(download_names_dataset(destination=args.output))
        return

    if args.command == "train":
        config = TrainingConfig(
            block_size=args.block_size,
            batch_size=args.batch_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            learning_rate=args.learning_rate,
            num_steps=args.num_steps,
            log_interval=args.log_interval,
            tokenizer_kind=args.tokenizer_kind,
            bpe_vocab_size=args.bpe_vocab_size,
        )
        documents = load_documents(args.dataset)
        artifacts = train_model(documents, config)
        if args.checkpoint_out is not None:
            save_checkpoint(
                path=args.checkpoint_out,
                model=artifacts.model,
                tokenizer=artifacts.tokenizer,
                training_config=config.__dict__.copy(),
            )
        sample = generate_text(
            artifacts.model,
            artifacts.tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"final_loss={artifacts.losses[-1]:.4f}")
        print(f"sample={sample}")
        if args.checkpoint_out is not None:
            print(f"checkpoint={args.checkpoint_out}")
        return

    if args.command == "generate":
        model, tokenizer, _ = load_checkpoint(args.checkpoint)
        sample = generate_text(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(sample)
