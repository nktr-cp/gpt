"""Dataset loading helpers for early experiments."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor

from .tokenizer import CharTokenizer


def load_documents(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_tokenizer(documents: list[str]) -> CharTokenizer:
    return CharTokenizer.fit(documents)


def encode_documents(
    documents: list[str],
    tokenizer: CharTokenizer,
    *,
    add_bos: bool = True,
    add_eos: bool = True,
) -> list[list[int]]:
    return [tokenizer.encode(document, add_bos=add_bos, add_eos=add_eos) for document in documents]


def flatten_token_sequences(sequences: list[list[int]]) -> list[int]:
    return [token_id for sequence in sequences for token_id in sequence]


def build_token_stream(
    documents: list[str],
    tokenizer: CharTokenizer,
    *,
    add_bos: bool = True,
    add_eos: bool = True,
) -> Tensor:
    encoded_documents = encode_documents(
        documents,
        tokenizer,
        add_bos=add_bos,
        add_eos=add_eos,
    )
    flattened = flatten_token_sequences(encoded_documents)
    return torch.tensor(flattened, dtype=torch.long)


def sample_next_token_batch(
    token_stream: Tensor,
    *,
    batch_size: int,
    block_size: int,
) -> tuple[Tensor, Tensor]:
    if token_stream.ndim != 1:
        msg = "token_stream must be a 1D tensor"
        raise ValueError(msg)
    if block_size <= 0:
        msg = "block_size must be positive"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch_size must be positive"
        raise ValueError(msg)
    if token_stream.size(0) <= block_size:
        msg = "token_stream must be longer than block_size"
        raise ValueError(msg)

    max_start = token_stream.size(0) - block_size - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([token_stream[start : start + block_size] for start in starts])
    y = torch.stack([token_stream[start + 1 : start + block_size + 1] for start in starts])
    return x, y


def save_tokenizer(tokenizer: CharTokenizer, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tokenizer.to_dict(), indent=2))
    return path


def load_tokenizer(path: Path) -> CharTokenizer:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        msg = "tokenizer payload must be a dictionary"
        raise TypeError(msg)
    return CharTokenizer.from_dict(payload)
