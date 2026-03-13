"""Training utilities for the GPT study project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .dataset import build_token_stream, build_tokenizer, load_documents, sample_next_token_batch
from .gpt import GPT, GPTConfig
from .tokenizer import Tokenizer


@dataclass(frozen=True)
class TrainingConfig:
    block_size: int = 32
    batch_size: int = 32
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    learning_rate: float = 1e-3
    num_steps: int = 200
    log_interval: int = 20
    tokenizer_kind: str = "bpe"
    bpe_vocab_size: int = 128

    def model_config(self, vocab_size: int) -> GPTConfig:
        return GPTConfig(
            vocab_size=vocab_size,
            block_size=self.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
        )


@dataclass(frozen=True)
class TrainingArtifacts:
    model: GPT
    tokenizer: Tokenizer
    token_stream: Tensor
    losses: list[float]


def compute_loss(logits: Tensor, targets: Tensor) -> Tensor:
    batch_size, sequence_length, vocab_size = logits.shape
    return F.cross_entropy(
        logits.view(batch_size * sequence_length, vocab_size),
        targets.view(batch_size * sequence_length),
    )


def train_model(documents: list[str], config: TrainingConfig) -> TrainingArtifacts:
    tokenizer = build_tokenizer(
        documents,
        tokenizer_kind=config.tokenizer_kind,
        vocab_size=config.bpe_vocab_size,
    )
    token_stream = build_token_stream(documents, tokenizer)
    model = GPT(config.model_config(tokenizer.vocab_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    losses: list[float] = []
    model.train()
    for step in range(config.num_steps):
        x, y = sample_next_token_batch(
            token_stream,
            batch_size=config.batch_size,
            block_size=config.block_size,
        )
        logits = model(x)
        loss = compute_loss(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % config.log_interval == 0 or step == config.num_steps - 1:
            print(f"step={step + 1} loss={loss.item():.4f}")

    return TrainingArtifacts(
        model=model,
        tokenizer=tokenizer,
        token_stream=token_stream,
        losses=losses,
    )


def train_model_from_path(dataset_path: str, config: TrainingConfig) -> TrainingArtifacts:
    documents = load_documents(Path(dataset_path))
    return train_model(documents, config)
