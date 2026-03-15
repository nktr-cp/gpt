"""Top-level GPT model wiring."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .attention import LayerCache
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
    positional_strategy: str = "learned"
    mlp_variant: str = "gelu"

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, int | str]) -> GPTConfig:
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
            positional_strategy=config.positional_strategy,
        )
        self.blocks = nn.ModuleList(
            TransformerBlock(
                n_embd=config.n_embd,
                n_head=config.n_head,
                expansion_factor=config.expansion_factor,
                positional_strategy=config.positional_strategy,
                mlp_variant=config.mlp_variant,
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

    def forward_with_cache(
        self,
        token_ids: Tensor,
        cache: list[LayerCache | None] | None,
        *,
        position_offset: int,
    ) -> tuple[Tensor, list[LayerCache]]:
        x = self.embedding.embed_with_positions(token_ids, position_offset=position_offset)
        next_cache: list[LayerCache] = []
        if cache is None:
            cache = [None] * len(self.blocks)

        for block, layer_cache in zip(self.blocks, cache, strict=False):
            x, updated_cache = block.forward_with_cache(
                x,
                cache=layer_cache,
                position_offset=position_offset,
            )
            next_cache.append(updated_cache)

        x = self.final_norm(x)
        return self.lm_head(x), next_cache

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_kv_cache: bool = True,
    ) -> Tensor:
        if temperature <= 0:
            msg = "temperature must be positive"
            raise ValueError(msg)

        if top_k is not None and top_k <= 0:
            msg = "top_k must be positive when provided"
            raise ValueError(msg)

        if use_kv_cache:
            return self.generate_with_cache(
                token_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        generated = token_ids
        block_size = self.embedding.block_size

        for _ in range(max_new_tokens):
            idx_cond = generated[:, -block_size:]
            logits = self(idx_cond)
            next_token_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                top_values, _ = torch.topk(
                    next_token_logits, k=min(top_k, next_token_logits.size(-1))
                )
                threshold = top_values[:, [-1]]
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < threshold, float("-inf")
                )
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

        return generated

    @torch.no_grad()
    def generate_with_cache(
        self,
        token_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        generated = token_ids
        cache: list[LayerCache | None] = [None] * len(self.blocks)
        logits: Tensor | None = None

        for position in range(token_ids.size(1)):
            current_token = token_ids[:, position : position + 1]
            logits, cache = self.forward_with_cache(
                current_token,
                cache,
                position_offset=position,
            )

        if logits is None:
            msg = "token_ids must contain at least one token"
            raise ValueError(msg)

        for _ in range(max_new_tokens):
            next_token_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                top_values, _ = torch.topk(
                    next_token_logits, k=min(top_k, next_token_logits.size(-1))
                )
                threshold = top_values[:, [-1]]
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < threshold, float("-inf")
                )
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            logits, cache = self.forward_with_cache(
                next_token,
                cache,
                position_offset=generated.size(1) - 1,
            )

        return generated
