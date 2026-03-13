"""Embedding components for the GPT study project."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GPTInputEmbedding(nn.Module):
    """Combine token embeddings and learned positional embeddings."""

    def __init__(
        self,
        *,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        positional_strategy: str = "learned",
    ) -> None:
        super().__init__()
        if positional_strategy not in {"learned", "rope"}:
            msg = "positional_strategy must be 'learned' or 'rope'"
            raise ValueError(msg)

        self.block_size = block_size
        self.n_embd = n_embd
        self.positional_strategy = positional_strategy
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        if positional_strategy == "learned":
            self.position_embedding: nn.Embedding | None = nn.Embedding(block_size, n_embd)
        else:
            self.position_embedding = None

    def embed_with_positions(self, token_ids: Tensor, *, position_offset: int = 0) -> Tensor:
        if token_ids.ndim != 2:
            msg = "token_ids must have shape (batch_size, sequence_length)"
            raise ValueError(msg)

        _, sequence_length = token_ids.shape
        end_position = position_offset + sequence_length
        if end_position > self.block_size:
            msg = f"sequence_length must be <= block_size ({self.block_size})"
            raise ValueError(msg)

        position_ids = torch.arange(
            position_offset,
            end_position,
            device=token_ids.device,
        )
        token_embeddings = self.token_embedding(token_ids)
        if self.position_embedding is None:
            return token_embeddings

        position_embeddings = self.position_embedding(position_ids).unsqueeze(0)
        return token_embeddings + position_embeddings

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.embed_with_positions(token_ids)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
