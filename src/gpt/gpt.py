"""Top-level GPT model wiring."""

from __future__ import annotations

from torch import Tensor, nn

from .blocks import RMSNorm, TransformerBlock
from .embeddings import GPTInputEmbedding


class GPT(nn.Module):
    """A minimal decoder-only GPT model."""

    def __init__(
        self,
        *,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        expansion_factor: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = GPTInputEmbedding(
            vocab_size=vocab_size,
            block_size=block_size,
            n_embd=n_embd,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(
                n_embd=n_embd,
                n_head=n_head,
                expansion_factor=expansion_factor,
            )
            for _ in range(n_layer)
        )
        self.final_norm = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, token_ids: Tensor) -> Tensor:
        x = self.embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)
