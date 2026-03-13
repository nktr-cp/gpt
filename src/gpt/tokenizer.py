"""Character-level tokenizer primitives."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialTokens:
    bos: str = "<bos>"
    eos: str = "<eos>"
    pad: str = "<pad>"
    unk: str = "<unk>"

    def as_list(self) -> list[str]:
        return [self.bos, self.eos, self.pad, self.unk]


class CharTokenizer:
    """A deterministic character-level tokenizer with reserved special tokens."""

    def __init__(self, vocab: list[str], special_tokens: SpecialTokens | None = None) -> None:
        self.special_tokens = special_tokens or SpecialTokens()
        self.vocab = vocab
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(vocab)}

    @classmethod
    def fit(
        cls,
        texts: list[str],
        special_tokens: SpecialTokens | None = None,
    ) -> CharTokenizer:
        resolved_special_tokens = special_tokens or SpecialTokens()
        base_vocab = sorted({char for text in texts for char in text})
        vocab = resolved_special_tokens.as_list() + base_vocab
        return cls(vocab=vocab, special_tokens=resolved_special_tokens)

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.special_tokens.bos]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.special_tokens.eos]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.special_tokens.pad]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.special_tokens.unk]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        token_ids = [self.token_to_id.get(char, self.unk_id) for char in text]
        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        chars: list[str] = []
        special_tokens = set(self.special_tokens.as_list())
        for token_id in token_ids:
            token = self.id_to_token[token_id]
            if skip_special_tokens and token in special_tokens:
                continue
            chars.append(token)
        return "".join(chars)

    def to_dict(self) -> dict[str, object]:
        return {
            "vocab": self.vocab,
            "special_tokens": {
                "bos": self.special_tokens.bos,
                "eos": self.special_tokens.eos,
                "pad": self.special_tokens.pad,
                "unk": self.special_tokens.unk,
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> CharTokenizer:
        special_tokens_payload = payload["special_tokens"]
        if not isinstance(special_tokens_payload, dict):
            msg = "special_tokens must be a dictionary"
            raise TypeError(msg)

        special_tokens = SpecialTokens(
            bos=str(special_tokens_payload["bos"]),
            eos=str(special_tokens_payload["eos"]),
            pad=str(special_tokens_payload["pad"]),
            unk=str(special_tokens_payload["unk"]),
        )
        vocab = payload["vocab"]
        if not isinstance(vocab, list):
            msg = "vocab must be a list"
            raise TypeError(msg)
        return cls(vocab=[str(token) for token in vocab], special_tokens=special_tokens)
