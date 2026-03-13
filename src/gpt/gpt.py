"""Top-level GPT model wiring."""

from __future__ import annotations

import torch
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

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> Tensor:
        if temperature <= 0:
            msg = "temperature must be positive"
            raise ValueError(msg)

        generated = token_ids
        block_size = self.embedding.block_size

        for _ in range(max_new_tokens):
            idx_cond = generated[:, -block_size:]
            logits = self(idx_cond)
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

        return generated
