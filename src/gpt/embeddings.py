"""Embedding components for the GPT study project."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GPTInputEmbedding(nn.Module):
    """Combine token embeddings and learned positional embeddings."""

    def __init__(self, *, vocab_size: int, block_size: int, n_embd: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            msg = "token_ids must have shape (batch_size, sequence_length)"
            raise ValueError(msg)

        _, sequence_length = token_ids.shape
        if sequence_length > self.block_size:
            msg = f"sequence_length must be <= block_size ({self.block_size})"
            raise ValueError(msg)

        position_ids = torch.arange(sequence_length, device=token_ids.device)
        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(position_ids).unsqueeze(0)
        return token_embeddings + position_embeddings

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
