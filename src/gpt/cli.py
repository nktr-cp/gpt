"""Project entrypoint for manual experiments."""

from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path

from .attention import SingleHeadCausalSelfAttention
from .data.download import dataset_path, download_names_dataset
from .dataset import build_token_stream, build_tokenizer, load_documents, sample_next_token_batch
from .embeddings import GPTInputEmbedding


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

    embedding_parser = subparsers.add_parser(
        "inspect-embeddings",
        help="Inspect token and positional embedding shapes.",
    )
    embedding_parser.add_argument(
        "--dataset",
        type=Path,
        default=dataset_path(),
        help="Path to the plain text dataset.",
    )
    embedding_parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of sampled sequences.",
    )
    embedding_parser.add_argument(
        "--block-size",
        type=int,
        default=8,
        help="Context length for each sampled sequence.",
    )
    embedding_parser.add_argument(
        "--n-embd",
        type=int,
        default=16,
        help="Embedding dimension.",
    )

    attention_parser = subparsers.add_parser(
        "inspect-attention",
        help="Inspect a single-head causal self-attention pass.",
    )
    attention_parser.add_argument(
        "--dataset",
        type=Path,
        default=dataset_path(),
        help="Path to the plain text dataset.",
    )
    attention_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of sampled sequences.",
    )
    attention_parser.add_argument(
        "--block-size",
        type=int,
        default=8,
        help="Context length for each sampled sequence.",
    )
    attention_parser.add_argument(
        "--n-embd",
        type=int,
        default=16,
        help="Embedding dimension.",
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
        return

    if args.command == "inspect-embeddings":
        documents = load_documents(args.dataset)
        tokenizer = build_tokenizer(documents)
        token_stream = build_token_stream(documents, tokenizer)
        x, _ = sample_next_token_batch(
            token_stream,
            batch_size=args.batch_size,
            block_size=args.block_size,
        )
        embedding = GPTInputEmbedding(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_embd=args.n_embd,
        )
        token_embeddings = embedding.token_embedding(x)
        position_ids = embedding.position_embedding.weight[: x.size(1)]
        combined = embedding(x)
        print(f"x_shape={tuple(x.shape)}")
        print(f"token_embeddings_shape={tuple(token_embeddings.shape)}")
        print(f"position_embeddings_shape={tuple(position_ids.shape)}")
        print(f"combined_shape={tuple(combined.shape)}")
        print(f"num_parameters={embedding.num_parameters}")
        return

    if args.command == "inspect-attention":
        documents = load_documents(args.dataset)
        tokenizer = build_tokenizer(documents)
        token_stream = build_token_stream(documents, tokenizer)
        x, _ = sample_next_token_batch(
            token_stream,
            batch_size=args.batch_size,
            block_size=args.block_size,
        )
        embedding = GPTInputEmbedding(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_embd=args.n_embd,
        )
        embedded = embedding(x)
        attention = SingleHeadCausalSelfAttention(n_embd=args.n_embd)
        inspected = attention.inspect(embedded)
        print(f"embedded_shape={tuple(embedded.shape)}")
        print(f"q_shape={tuple(inspected['q'].shape)}")
        print(f"scores_shape={tuple(inspected['scores'].shape)}")
        print(f"attention_weights_shape={tuple(inspected['attention_weights'].shape)}")
        print(f"output_shape={tuple(inspected['output'].shape)}")
        print(f"mask_row_0={inspected['mask'][0].int().tolist()}")
        print(f"mask_last_row={inspected['mask'][-1].int().tolist()}")
        print(f"attention_row_0={inspected['attention_weights'][0, 0].tolist()}")
        print(f"attention_last_row={inspected['attention_weights'][0, -1].tolist()}")
        print(f"num_parameters={attention.num_parameters}")
