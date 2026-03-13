"""Dataset loading helpers for early experiments."""

from __future__ import annotations

import json
from pathlib import Path

from .tokenizer import CharTokenizer


def load_documents(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_tokenizer(documents: list[str]) -> CharTokenizer:
    return CharTokenizer.fit(documents)


def save_tokenizer(tokenizer: CharTokenizer, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tokenizer.to_dict(), indent=2))
    return path


def load_tokenizer(path: Path) -> CharTokenizer:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        msg = "tokenizer payload must be a dictionary"
        raise TypeError(msg)
    return CharTokenizer.from_dict(payload)
