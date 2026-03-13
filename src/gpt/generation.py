"""Autoregressive text generation helpers."""

from __future__ import annotations

import torch

from .gpt import GPT
from .tokenizer import Tokenizer


def generate_text(
    model: GPT,
    tokenizer: Tokenizer,
    *,
    prompt: str = "",
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int | None = None,
    use_kv_cache: bool = True,
) -> str:
    model.eval()

    if prompt:
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
    else:
        prompt_ids = [tokenizer.bos_id]

    x = torch.tensor([prompt_ids], dtype=torch.long)
    generated = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        use_kv_cache=use_kv_cache,
    )[0].tolist()

    continuation = generated[1:]
    if tokenizer.eos_id in continuation:
        continuation = continuation[: continuation.index(tokenizer.eos_id)]
    return tokenizer.decode(continuation)
