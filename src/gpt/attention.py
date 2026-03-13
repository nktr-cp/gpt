"""Attention components for the GPT study project."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

LayerCache = tuple[Tensor, Tensor]


def causal_mask(sequence_length: int, device: torch.device) -> Tensor:
    return torch.tril(torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool))


class SingleHeadCausalSelfAttention(nn.Module):
    """A single masked self-attention head for decoder-only GPT."""

    def __init__(self, *, n_embd: int) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.inspect(x)["output"]

    def inspect(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 3:
            msg = "x must have shape (batch_size, sequence_length, n_embd)"
            raise ValueError(msg)
        if x.size(-1) != self.n_embd:
            msg = f"last dimension must be n_embd ({self.n_embd})"
            raise ValueError(msg)

        _, sequence_length, _ = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scale = 1.0 / math.sqrt(self.n_embd)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        mask = causal_mask(sequence_length, x.device)
        masked_scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
        attention_weights = torch.softmax(masked_scores, dim=-1)
        output = torch.matmul(attention_weights, v)

        return {
            "q": q,
            "k": k,
            "v": v,
            "scores": scores,
            "mask": mask,
            "masked_scores": masked_scores,
            "attention_weights": attention_weights,
            "output": output,
        }

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class MultiHeadCausalSelfAttention(nn.Module):
    """A minimal multi-head masked self-attention module for decoder-only GPT."""

    def __init__(self, *, n_embd: int, n_head: int) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            msg = "n_embd must be divisible by n_head"
            raise ValueError(msg)

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, _ = x.shape
        x = x.view(batch_size, sequence_length, self.n_head, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch_size, _, sequence_length, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, sequence_length, self.n_embd)

    def forward(self, x: Tensor) -> Tensor:
        return self.inspect(x)["output"]

    def forward_with_cache(
        self,
        x: Tensor,
        cache: LayerCache | None = None,
    ) -> tuple[Tensor, LayerCache]:
        if x.ndim != 3:
            msg = "x must have shape (batch_size, sequence_length, n_embd)"
            raise ValueError(msg)
        if x.size(1) != 1:
            msg = "forward_with_cache expects a single new token with shape (batch_size, 1, n_embd)"
            raise ValueError(msg)

        q = self._split_heads(self.query(x))
        k_new = self._split_heads(self.key(x))
        v_new = self._split_heads(self.value(x))

        if cache is None:
            k_all = k_new
            v_all = v_new
        else:
            past_k, past_v = cache
            k_all = torch.cat((past_k, k_new), dim=2)
            v_all = torch.cat((past_v, v_new), dim=2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k_all.transpose(-2, -1)) * scale
        attention_weights = torch.softmax(scores, dim=-1)
        head_outputs = torch.matmul(attention_weights, v_all)
        merged = self._merge_heads(head_outputs)
        return self.proj(merged), (k_all, v_all)

    def inspect(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 3:
            msg = "x must have shape (batch_size, sequence_length, n_embd)"
            raise ValueError(msg)
        if x.size(-1) != self.n_embd:
            msg = f"last dimension must be n_embd ({self.n_embd})"
            raise ValueError(msg)

        _, sequence_length, _ = x.shape
        q = self._split_heads(self.query(x))
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = causal_mask(sequence_length, x.device).unsqueeze(0).unsqueeze(0)
        masked_scores = scores.masked_fill(~mask, float("-inf"))
        attention_weights = torch.softmax(masked_scores, dim=-1)
        head_outputs = torch.matmul(attention_weights, v)
        merged = self._merge_heads(head_outputs)
        output = self.proj(merged)

        return {
            "q": q,
            "k": k,
            "v": v,
            "scores": scores,
            "mask": mask,
            "masked_scores": masked_scores,
            "attention_weights": attention_weights,
            "head_outputs": head_outputs,
            "merged": merged,
            "output": output,
        }

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
