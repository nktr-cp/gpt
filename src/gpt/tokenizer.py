"""Tokenizer primitives."""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SpecialTokens:
    bos: str = "<bos>"
    eos: str = "<eos>"
    pad: str = "<pad>"
    unk: str = "<unk>"

    def as_list(self) -> list[str]:
        return [self.bos, self.eos, self.pad, self.unk]


class Tokenizer(Protocol):
    special_tokens: SpecialTokens
    vocab: list[str]
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    @property
    def pad_id(self) -> int: ...

    @property
    def unk_id(self) -> int: ...

    @property
    def vocab_size(self) -> int: ...

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]: ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str: ...

    def to_dict(self) -> dict[str, object]: ...


class BaseTokenizer:
    """Shared tokenizer helpers."""

    kind = "base"

    def __init__(self, vocab: list[str], special_tokens: SpecialTokens | None = None) -> None:
        self.special_tokens = special_tokens or SpecialTokens()
        self.vocab = vocab
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(vocab)}

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

    def _with_special_tokens(
        self,
        token_ids: list[int],
        *,
        add_bos: bool,
        add_eos: bool,
    ) -> list[int]:
        output = list(token_ids)
        if add_bos:
            output.insert(0, self.bos_id)
        if add_eos:
            output.append(self.eos_id)
        return output

    def _decode_tokens(self, tokens: list[str], *, skip_special_tokens: bool) -> str:
        special_tokens = set(self.special_tokens.as_list())
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in special_tokens]
        return "".join(tokens)

    def _special_tokens_dict(self) -> dict[str, str]:
        return {
            "bos": self.special_tokens.bos,
            "eos": self.special_tokens.eos,
            "pad": self.special_tokens.pad,
            "unk": self.special_tokens.unk,
        }


class CharTokenizer(BaseTokenizer):
    """A deterministic character-level tokenizer with reserved special tokens."""

    kind = "char"

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

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        token_ids = [self.token_to_id.get(char, self.unk_id) for char in text]
        return self._with_special_tokens(token_ids, add_bos=add_bos, add_eos=add_eos)

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        tokens = [self.id_to_token[token_id] for token_id in token_ids]
        return self._decode_tokens(tokens, skip_special_tokens=skip_special_tokens)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "vocab": self.vocab,
            "special_tokens": self._special_tokens_dict(),
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


TokenSequence = tuple[str, ...]
Pair = tuple[str, str]
SequenceCounts = Counter[TokenSequence]


def count_pairs(token_sequences: list[list[str]]) -> Counter[Pair]:
    counts: Counter[tuple[str, str]] = Counter()
    for sequence in token_sequences:
        for left, right in zip(sequence, sequence[1:], strict=False):
            counts[(left, right)] += 1
    return counts


def count_pairs_in_sequences(sequence_counts: SequenceCounts) -> Counter[Pair]:
    counts: Counter[Pair] = Counter()
    for sequence, frequency in sequence_counts.items():
        for left, right in zip(sequence, sequence[1:], strict=False):
            counts[(left, right)] += frequency
    return counts


def merge_pair(sequence: list[str], pair: Pair) -> list[str]:
    merged: list[str] = []
    i = 0
    while i < len(sequence):
        if i < len(sequence) - 1 and (sequence[i], sequence[i + 1]) == pair:
            merged.append(sequence[i] + sequence[i + 1])
            i += 2
            continue
        merged.append(sequence[i])
        i += 1
    return merged


def merge_pair_in_sequence(sequence: TokenSequence, pair: Pair) -> TokenSequence:
    return tuple(merge_pair(list(sequence), pair))


def merge_pair_in_sequences(sequence_counts: SequenceCounts, pair: Pair) -> SequenceCounts:
    merged_counts: SequenceCounts = Counter()
    cache: dict[TokenSequence, TokenSequence] = {}

    for sequence, frequency in sequence_counts.items():
        merged_sequence = cache.get(sequence)
        if merged_sequence is None:
            merged_sequence = merge_pair_in_sequence(sequence, pair)
            cache[sequence] = merged_sequence
        merged_counts[merged_sequence] += frequency

    return merged_counts


def pair_counts_in_sequence(sequence: TokenSequence) -> Counter[Pair]:
    counts: Counter[Pair] = Counter()
    for left, right in zip(sequence, sequence[1:], strict=False):
        counts[(left, right)] += 1
    return counts


def initialize_pair_index(
    sequence_counts: SequenceCounts,
) -> tuple[Counter[Pair], dict[Pair, set[TokenSequence]], list[tuple[int, Pair]]]:
    pair_counts: Counter[Pair] = Counter()
    pair_to_sequences: dict[Pair, set[TokenSequence]] = defaultdict(set)

    for sequence, frequency in sequence_counts.items():
        sequence_pair_counts = pair_counts_in_sequence(sequence)
        for pair, count in sequence_pair_counts.items():
            pair_counts[pair] += count * frequency
            pair_to_sequences[pair].add(sequence)

    heap: list[tuple[int, Pair]] = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)
    return pair_counts, pair_to_sequences, heap


def pop_best_pair(
    pair_counts: Counter[Pair],
    heap: list[tuple[int, Pair]],
) -> tuple[Pair, int] | None:
    while heap:
        neg_count, pair = heapq.heappop(heap)
        current_count = pair_counts.get(pair, 0)
        if current_count == 0:
            continue
        if -neg_count != current_count:
            continue
        return pair, current_count
    return None


def apply_bpe_merge(
    pair: Pair,
    *,
    sequence_counts: SequenceCounts,
    pair_counts: Counter[Pair],
    pair_to_sequences: dict[Pair, set[TokenSequence]],
    heap: list[tuple[int, Pair]],
) -> None:
    impacted_sequences = list(pair_to_sequences.get(pair, set()))
    if not impacted_sequences:
        pair_counts.pop(pair, None)
        return

    produced_sequences: SequenceCounts = Counter()
    changed_pairs: set[Pair] = set()

    for sequence in impacted_sequences:
        frequency = sequence_counts.pop(sequence)
        old_pair_counts = pair_counts_in_sequence(sequence)
        for old_pair, count in old_pair_counts.items():
            updated_count = pair_counts[old_pair] - (count * frequency)
            if updated_count > 0:
                pair_counts[old_pair] = updated_count
            else:
                pair_counts.pop(old_pair, None)
            pair_to_sequences[old_pair].discard(sequence)
            if not pair_to_sequences[old_pair]:
                pair_to_sequences.pop(old_pair, None)
            changed_pairs.add(old_pair)

        merged_sequence = merge_pair_in_sequence(sequence, pair)
        produced_sequences[merged_sequence] += frequency

    for sequence, frequency in produced_sequences.items():
        sequence_counts[sequence] += frequency
        new_pair_counts = pair_counts_in_sequence(sequence)
        for new_pair, count in new_pair_counts.items():
            pair_counts[new_pair] += count * frequency
            pair_to_sequences.setdefault(new_pair, set()).add(sequence)
            changed_pairs.add(new_pair)

    for changed_pair in changed_pairs:
        count = pair_counts.get(changed_pair, 0)
        if count > 0:
            heapq.heappush(heap, (-count, changed_pair))


class BPETokenizer(BaseTokenizer):
    """A simple character-initialized BPE tokenizer."""

    kind = "bpe"

    def __init__(
        self,
        vocab: list[str],
        merges: list[tuple[str, str]],
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        super().__init__(vocab=vocab, special_tokens=special_tokens)
        self.merges = merges

    @classmethod
    def fit(
        cls,
        texts: list[str],
        *,
        vocab_size: int,
        special_tokens: SpecialTokens | None = None,
    ) -> BPETokenizer:
        resolved_special_tokens = special_tokens or SpecialTokens()
        alphabet = sorted({char for text in texts for char in text})
        vocab = resolved_special_tokens.as_list() + alphabet
        if vocab_size < len(vocab):
            msg = "vocab_size is too small for the special tokens and alphabet"
            raise ValueError(msg)

        sequence_counts: SequenceCounts = Counter(tuple(text) for text in texts)
        pair_counts, pair_to_sequences, heap = initialize_pair_index(sequence_counts)
        merges: list[tuple[str, str]] = []

        while len(vocab) < vocab_size:
            best = pop_best_pair(pair_counts, heap)
            if best is None:
                break

            best_pair, best_count = best
            if best_count < 2:
                break

            merged_token = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            apply_bpe_merge(
                best_pair,
                sequence_counts=sequence_counts,
                pair_counts=pair_counts,
                pair_to_sequences=pair_to_sequences,
                heap=heap,
            )
            if merged_token not in vocab:
                vocab.append(merged_token)

        return cls(vocab=vocab, merges=merges, special_tokens=resolved_special_tokens)

    def _apply_merges(self, tokens: list[str]) -> list[str]:
        merged = list(tokens)
        for pair in self.merges:
            merged = merge_pair(merged, pair)
        return merged

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        merged_tokens = self._apply_merges(list(text))
        token_ids = [self.token_to_id.get(token, self.unk_id) for token in merged_tokens]
        return self._with_special_tokens(token_ids, add_bos=add_bos, add_eos=add_eos)

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        tokens = [self.id_to_token[token_id] for token_id in token_ids]
        return self._decode_tokens(tokens, skip_special_tokens=skip_special_tokens)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "vocab": self.vocab,
            "special_tokens": self._special_tokens_dict(),
            "merges": [[left, right] for left, right in self.merges],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> BPETokenizer:
        special_tokens_payload = payload["special_tokens"]
        if not isinstance(special_tokens_payload, dict):
            msg = "special_tokens must be a dictionary"
            raise TypeError(msg)
        vocab = payload["vocab"]
        merges = payload["merges"]
        if not isinstance(vocab, list):
            msg = "vocab must be a list"
            raise TypeError(msg)
        if not isinstance(merges, list):
            msg = "merges must be a list"
            raise TypeError(msg)

        special_tokens = SpecialTokens(
            bos=str(special_tokens_payload["bos"]),
            eos=str(special_tokens_payload["eos"]),
            pad=str(special_tokens_payload["pad"]),
            unk=str(special_tokens_payload["unk"]),
        )
        parsed_merges = [(str(left), str(right)) for left, right in merges]
        return cls(
            vocab=[str(token) for token in vocab],
            merges=parsed_merges,
            special_tokens=special_tokens,
        )


def tokenizer_from_dict(payload: dict[str, object]) -> Tokenizer:
    kind = payload.get("kind", "char")
    if kind == CharTokenizer.kind:
        return CharTokenizer.from_dict(payload)
    if kind == BPETokenizer.kind:
        return BPETokenizer.from_dict(payload)
    msg = f"unsupported tokenizer kind: {kind}"
    raise ValueError(msg)
