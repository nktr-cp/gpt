"""Top-level GPT model wiring."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .blocks import RMSNorm, TransformerBlock
from .embeddings import GPTInputEmbedding


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    expansion_factor: int = 4

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, int]) -> GPTConfig:
        return cls(**payload)


class GPT(nn.Module):
    """A minimal decoder-only GPT model."""

    def __init__(
        self,
        config: GPTConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.embedding = GPTInputEmbedding(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            n_embd=config.n_embd,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(
                n_embd=config.n_embd,
                n_head=config.n_head,
                expansion_factor=config.expansion_factor,
            )
            for _ in range(config.n_layer)
        )
        self.final_norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

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
