"""Lightweight evaluation utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .dataset import sample_next_token_batch
from .gpt import GPT
from .training import compute_loss


@dataclass(frozen=True)
class EvaluationMetrics:
    loss: float
    perplexity: float


def split_token_stream(
    token_stream: torch.Tensor,
    *,
    validation_fraction: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 < validation_fraction < 1:
        msg = "validation_fraction must be between 0 and 1"
        raise ValueError(msg)
    split_index = max(2, int(token_stream.size(0) * (1 - validation_fraction)))
    return token_stream[:split_index], token_stream[split_index:]


@torch.no_grad()
def evaluate_model(
    model: GPT,
    token_stream: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    num_batches: int = 20,
) -> EvaluationMetrics:
    model.eval()
    losses: list[float] = []
    for _ in range(num_batches):
        x, y = sample_next_token_batch(
            token_stream,
            batch_size=batch_size,
            block_size=block_size,
        )
        logits = model(x)
        losses.append(compute_loss(logits, y).item())

    loss = sum(losses) / len(losses)
    return EvaluationMetrics(loss=loss, perplexity=math.exp(loss))
