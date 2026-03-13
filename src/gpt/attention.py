"""Attention components for the GPT study project."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


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
