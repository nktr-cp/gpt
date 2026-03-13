"""Rotary position embedding helpers."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def rotate_half(x: Tensor) -> Tensor:
    """Rotate pairs of features by 90 degrees."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(start_dim=-2)


class RotaryEmbedding(nn.Module):
    """Apply deterministic rotary position encoding to query/key tensors."""

    def __init__(self, dim: int, *, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            msg = "RoPE head dimension must be even"
            raise ValueError(msg)

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.dim = dim
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def angles(
        self,
        sequence_length: int,
        *,
        device: torch.device,
        position_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        positions = torch.arange(
            position_offset,
            position_offset + sequence_length,
            device=device,
            dtype=self.inv_freq.dtype,
        )
        angles = torch.outer(positions, self.inv_freq.to(device))
        cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1)
        sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1)
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

    def apply(self, x: Tensor, *, position_offset: int = 0) -> Tensor:
        if x.size(-1) != self.dim:
            msg = f"last dimension must be rotary dim ({self.dim})"
            raise ValueError(msg)

        cos, sin = self.angles(
            x.size(-2),
            device=x.device,
            position_offset=position_offset,
        )
        return (x * cos) + (rotate_half(x) * sin)
