"""Transformer block components for the GPT study project."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .attention import LayerCache, MultiHeadCausalSelfAttention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization without mean subtraction."""

    def __init__(self, n_embd: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(rms + self.eps)
        return normalized * self.weight


class FeedForward(nn.Module):
    """A position-wise feed-forward network."""

    def __init__(self, *, n_embd: int, expansion_factor: int = 4) -> None:
        super().__init__()
        hidden_dim = expansion_factor * n_embd
        self.in_proj = nn.Linear(n_embd, hidden_dim, bias=False)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(hidden_dim, n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_proj(self.activation(self.in_proj(x)))


class TransformerBlock(nn.Module):
    """A pre-norm decoder block with attention and feed-forward sublayers."""

    def __init__(
        self,
        *,
        n_embd: int,
        n_head: int,
        expansion_factor: int = 4,
        positional_strategy: str = "learned",
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(n_embd)
        self.attention = MultiHeadCausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            use_rope=positional_strategy == "rope",
        )
        self.feed_forward_norm = RMSNorm(n_embd)
        self.feed_forward = FeedForward(
            n_embd=n_embd,
            expansion_factor=expansion_factor,
        )

    def forward(self, x: Tensor, *, position_offset: int = 0) -> Tensor:
        x = x + self.attention(self.attention_norm(x), position_offset=position_offset)
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x

    def forward_with_cache(
        self,
        x: Tensor,
        cache: LayerCache | None = None,
        *,
        position_offset: int = 0,
    ) -> tuple[Tensor, LayerCache]:
        attention_out, next_cache = self.attention.forward_with_cache(
            self.attention_norm(x),
            cache=cache,
            position_offset=position_offset,
        )
        x = x + attention_out
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x, next_cache
