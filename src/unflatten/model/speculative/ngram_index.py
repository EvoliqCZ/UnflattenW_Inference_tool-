"""Suffix / n-gram continuation indices for training-free drafting
(`docs/plans/drafter-research-2026-09-28.md` idea 3, 2026-09-28).

Two sources, same question -- "what came next the last time these `L`
tokens appeared?", longest `L` first (`max_len` down to `min_len`):

- :class:`PageNgramIndex`: the page's own committed tokens, grown
  incrementally as the decode commits; the MOST RECENT continuation wins
  (prompt-lookup style -- a table row repeats the row above it).
- :class:`CorpusNgramIndex`: count-based over many token streams (e.g. the
  training split's gold streams); the MOST FREQUENT continuation wins
  (DOMINO-style), ties to the lowest id. Contexts are 64-bit polynomial
  hashes, one sorted table per length, so a 50 M-token corpus fits in a
  few GB and a lookup is one `searchsorted`. A hash collision can only
  produce a wrong DRAFT, never a wrong output: verification is exact.

Neither index knows the grammar; `grammar_draft.GrammarWalkDrafter`
filters what they propose.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

_MASK = (1 << 64) - 1
#: Odd 64-bit multiplier (golden-ratio constant); arithmetic wraps mod 2**64.
_P = 0x9E3779B97F4A7C15


class PageNgramIndex:
    """Most-recent continuation of every context of length `min_len..max_len`
    seen in one page's committed tokens. `extend_to` is fed the whole
    committed prefix each round and only indexes what is new."""

    def __init__(self, *, min_len: int = 2, max_len: int = 8) -> None:
        if not 1 <= min_len <= max_len:
            raise ValueError(f"need 1 <= min_len <= max_len, got {min_len}, {max_len}")
        self.min_len = min_len
        self.max_len = max_len
        self._tokens: list[int] = []
        self._tables: dict[int, dict[tuple[int, ...], int]] = {
            length: {} for length in range(min_len, max_len + 1)
        }

    def extend_to(self, committed: Sequence[int]) -> None:
        if len(committed) < len(self._tokens):
            raise ValueError("committed prefix shrank: one PageNgramIndex per decode")
        tokens = self._tokens
        for j in range(len(tokens), len(committed)):
            token = int(committed[j])
            for length in range(self.min_len, min(self.max_len, j) + 1):
                self._tables[length][tuple(tokens[j - length : j])] = token
            tokens.append(token)

    def predict(self, context: Sequence[int]) -> tuple[int, int] | None:
        """`(next_token, match_length)` for the longest indexed suffix of
        `context`, or `None`."""
        for length in range(min(self.max_len, len(context)), self.min_len - 1, -1):
            token = self._tables[length].get(tuple(context[-length:]))
            if token is not None:
                return token, length
        return None


def _suffix_hashes(context: Sequence[int], max_len: int) -> list[int]:
    """`[h_1, ..., h_m]`, `h_L` the hash of the last `L` tokens of `context`
    under the same polynomial :func:`build_corpus_ngram_index` vectorizes
    (`h = sum (t_j + 1) * P**(L-1-j)` over the window, mod 2**64)."""
    hashes: list[int] = []
    h = 0
    power = 1
    for token in reversed(context[-max_len:]):
        h = (h + (int(token) + 1) * power) & _MASK
        power = (power * _P) & _MASK
        hashes.append(h)
    return hashes


class CorpusNgramIndex:
    """Count-based continuation tables; build with
    :func:`build_corpus_ngram_index`, persist with :meth:`save`/:meth:`load`."""

    def __init__(
        self,
        *,
        min_len: int,
        max_len: int,
        tables: Mapping[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        page_ids: tuple[str, ...] = (),
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        self.min_len = min_len
        self.max_len = max_len
        #: length -> (sorted uint64 context hashes, best next token,
        #: its count, total count of the context)
        self.tables = dict(tables)
        self.page_ids = tuple(page_ids)
        self.meta = dict(meta or {})

    def predict(self, context: Sequence[int]) -> tuple[int, int] | None:
        hit = self.lookup(context)
        return None if hit is None else (hit[0], hit[1])

    def lookup(self, context: Sequence[int]) -> tuple[int, int, int, int] | None:
        """`(next_token, match_length, count, total)` for the longest
        indexed suffix of `context`, or `None`."""
        hashes = _suffix_hashes(context, self.max_len)
        for length in range(min(self.max_len, len(hashes)), self.min_len - 1, -1):
            keys, nxt, count, total = self.tables[length]
            key = np.uint64(hashes[length - 1])
            i = int(np.searchsorted(keys, key))
            if i < len(keys) and keys[i] == key:
                return int(nxt[i]), length, int(count[i]), int(total[i])
        return None

    def save(self, path: Path) -> None:
        arrays: dict[str, np.ndarray] = {}
        for length, (keys, nxt, count, total) in self.tables.items():
            arrays[f"keys_{length}"] = keys
            arrays[f"next_{length}"] = nxt
            arrays[f"count_{length}"] = count
            arrays[f"total_{length}"] = total
        header = {"min_len": self.min_len, "max_len": self.max_len, "meta": self.meta}
        arrays["header"] = np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8)
        arrays["page_ids"] = np.array(self.page_ids, dtype=np.str_)
        with path.open("wb") as handle:
            np.savez(handle, **arrays)

    @classmethod
    def load(cls, path: Path) -> CorpusNgramIndex:
        with np.load(path, allow_pickle=False) as payload:
            header = json.loads(payload["header"].tobytes().decode("utf-8"))
            tables = {
                length: (
                    payload[f"keys_{length}"],
                    payload[f"next_{length}"],
                    payload[f"count_{length}"],
                    payload[f"total_{length}"],
                )
                for length in range(header["min_len"], header["max_len"] + 1)
            }
            page_ids = tuple(str(p) for p in payload["page_ids"])
        return cls(
            min_len=header["min_len"], max_len=header["max_len"], tables=tables,
            page_ids=page_ids, meta=header["meta"],
        )


def _mode_table(
    keys: np.ndarray, nxt: np.ndarray, min_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per distinct key: the most frequent next token (ties -> lowest id),
    its count and the key's total count; keys with a best count below
    `min_count` are dropped. Output sorted by key."""
    if len(keys) == 0:
        empty = np.empty(0, dtype=np.int32)
        return np.empty(0, dtype=np.uint64), empty, empty.copy(), empty.copy()
    order = np.lexsort((nxt, keys))
    keys, nxt = keys[order], nxt[order]
    new_pair = np.empty(len(keys), dtype=bool)
    new_pair[0] = True
    new_pair[1:] = (keys[1:] != keys[:-1]) | (nxt[1:] != nxt[:-1])
    starts = np.flatnonzero(new_pair)
    counts = np.diff(np.append(starts, len(keys)))
    pair_keys, pair_next = keys[starts], nxt[starts]
    del keys, nxt, order
    new_key = np.empty(len(pair_keys), dtype=bool)
    new_key[0] = True
    new_key[1:] = pair_keys[1:] != pair_keys[:-1]
    key_starts = np.flatnonzero(new_key)
    totals = np.add.reduceat(counts, key_starts)
    # Best pair per key: sort by (key, -count, next); the first of each key wins.
    order = np.lexsort((pair_next, -counts, pair_keys))
    pair_keys, pair_next, counts = pair_keys[order], pair_next[order], counts[order]
    first = np.flatnonzero(np.append(True, pair_keys[1:] != pair_keys[:-1]))
    best_keys, best_next, best_count = pair_keys[first], pair_next[first], counts[first]
    keep = best_count >= min_count
    return (
        best_keys[keep],
        best_next[keep].astype(np.int32),
        best_count[keep].astype(np.int32),
        totals[keep].astype(np.int32),
    )


def build_corpus_ngram_index(
    streams: Iterable[Sequence[int]],
    *,
    min_len: int = 2,
    max_len: int = 8,
    min_count: int = 1,
    page_ids: Sequence[str] = (),
    meta: Mapping[str, Any] | None = None,
) -> CorpusNgramIndex:
    """Count every (context of length `L`, next token) pair inside each
    stream -- a context never spans two streams -- for `L` in
    `min_len..max_len`, and keep the mode per context."""
    if not 1 <= min_len <= max_len:
        raise ValueError(f"need 1 <= min_len <= max_len, got {min_len}, {max_len}")
    parts = [np.asarray(stream, dtype=np.int64) for stream in streams]
    parts = [p for p in parts if len(p)]
    if parts:
        tokens = np.concatenate(parts)
        lengths = np.array([len(p) for p in parts], dtype=np.int64)
    else:
        tokens = np.empty(0, dtype=np.int64)
        lengths = np.empty(0, dtype=np.int64)
    starts = np.repeat(np.cumsum(lengths) - lengths, lengths)
    ends = np.repeat(np.cumsum(lengths), lengths)
    positions = np.arange(len(tokens), dtype=np.int64)
    offset = positions - starts  # index of each token within its stream
    has_next = positions + 1 < ends
    values = (tokens + 1).astype(np.uint64)
    multiplier = np.uint64(_P)
    tables: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    # h_L[i]: hash of the window of length L ending at position i.
    h = values.copy()
    for length in range(1, max_len + 1):
        if length > 1:
            shifted = np.zeros_like(h)
            shifted[1:] = h[:-1]
            with np.errstate(over="ignore"):
                h = shifted * multiplier + values
        if length < min_len:
            continue
        valid = np.flatnonzero((offset >= length - 1) & has_next)
        tables[length] = _mode_table(h[valid], tokens[valid + 1], min_count)
    return CorpusNgramIndex(
        min_len=min_len, max_len=max_len, tables=tables, page_ids=tuple(page_ids), meta=meta
    )
