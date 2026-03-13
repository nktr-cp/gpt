"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .gpt import GPT, GPTConfig
from .tokenizer import Tokenizer, tokenizer_from_dict


def save_checkpoint(
    *,
    path: Path,
    model: GPT,
    tokenizer: Tokenizer,
    training_config: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "training_config": training_config,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path) -> tuple[GPT, Tokenizer, dict[str, Any]]:
    payload = torch.load(path, weights_only=False)
    model_config = GPTConfig.from_dict(payload["model_config"])
    model = GPT(model_config)
    model.load_state_dict(payload["model_state_dict"])
    tokenizer = tokenizer_from_dict(payload["tokenizer"])
    training_config = dict(payload["training_config"])
    return model, tokenizer, training_config
