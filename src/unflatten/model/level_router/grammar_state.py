"""Grammar state the level router is conditioned on (RFC007 §6.2).

The router needs to know *where in the AST* the decoder is when it emits a
token: opening a table, inside a cell, inside a paragraph's text run, and
so on. That is read deterministically off the composite token stream -- the
71-token structural alphabet (`events_vocab`) -- with a container stack:

- `<X>_OPEN` and `RESUME_NODE(x)` push `x`;
- `<X>_CLOSE` and `SUSPEND_NODE(x)` pop back to (and including) the
  innermost open `x`; an unmatched close changes nothing (the tracker is
  total on any id sequence -- free-running decode is grammar-constrained,
  but this module must not be what raises on a malformed prefix);
- `TEXT_BEGIN` / `TEXT_END` toggle `in_text`;
- every other id (Qwen text, bytes, leaves) leaves the state unchanged.

The state id a position gets is the state AFTER consuming that position's
input token -- exactly what the decoder hidden state at that position has
seen, so conditioning is causal by construction.

`state_id = (bucket * 2 + in_text) * 2 + in_table`, where `bucket` is
`1 + index` of the nearest enclosing container listed in the policy's
`state_containers` (0 when none is open). "Heading" is not an RFC004
container -- a heading is a paragraph whose style is a typed property
predicted later -- so it can only reach the router through the hidden
state, not through this id (RFC007 §4.2, stated limitation).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType

from unflatten.model import events_vocab
from unflatten.model.composite_vocab import PackedRanges
from unflatten.model.level_router.policy import LevelRouterPolicy


def _invert(mapping: dict[str, str] | MappingProxyType) -> dict[int, str]:
    return {events_vocab.TOKEN_IDS[name]: kind for kind, name in mapping.items()}


_PUSH: dict[int, str] = {**_invert(events_vocab.OPEN_TOKEN), **_invert(events_vocab.RESUME_TOKEN)}
_POP: dict[int, str] = {**_invert(events_vocab.CLOSE_TOKEN), **_invert(events_vocab.SUSPEND_TOKEN)}
_TEXT_BEGIN = events_vocab.TOKEN_IDS[events_vocab.TEXT_BEGIN]
_TEXT_END = events_vocab.TOKEN_IDS[events_vocab.TEXT_END]


def num_grammar_states(policy: LevelRouterPolicy) -> int:
    return (len(policy.state_containers) + 1) * 4


@dataclass(frozen=True, slots=True)
class GrammarState:
    stack: tuple[str, ...]
    in_text: bool

    @classmethod
    def initial(cls) -> GrammarState:
        return cls(stack=(), in_text=False)

    @property
    def innermost(self) -> str | None:
        return self.stack[-1] if self.stack else None

    @property
    def in_table(self) -> bool:
        return "table" in self.stack

    def advance(self, packed_id: int, ranges: PackedRanges) -> GrammarState:
        local = packed_id - ranges.structural_start
        if not 0 <= local < ranges.structural_count:
            return self
        if local in _PUSH:
            return GrammarState(stack=(*self.stack, _PUSH[local]), in_text=self.in_text)
        if local in _POP:
            kind = _POP[local]
            if kind not in self.stack:
                return self
            cut = len(self.stack) - 1 - self.stack[::-1].index(kind)
            return GrammarState(stack=self.stack[:cut], in_text=self.in_text)
        if local == _TEXT_BEGIN:
            return GrammarState(stack=self.stack, in_text=True)
        if local == _TEXT_END:
            return GrammarState(stack=self.stack, in_text=False)
        return self

    def state_id(self, policy: LevelRouterPolicy) -> int:
        bucket = 0
        for kind in reversed(self.stack):
            if kind in policy.state_containers:
                bucket = 1 + policy.state_containers.index(kind)
                break
        return (bucket * 2 + int(self.in_text)) * 2 + int(self.in_table)


def grammar_state_ids(
    packed_ids: Sequence[int], ranges: PackedRanges, policy: LevelRouterPolicy
) -> list[int]:
    """One state id per position of one sequence (the teacher-forced path)."""
    state = GrammarState.initial()
    out: list[int] = []
    for packed_id in packed_ids:
        state = state.advance(int(packed_id), ranges)
        out.append(state.state_id(policy))
    return out


@dataclass(frozen=True, slots=True)
class GrammarTrace:
    """Persistent (linked) grammar-state history for incremental decode.

    `IncrementalDecodeState` carries one of these so `encode_step` can
    advance by one token in O(1) and `crop`/`rollback_to` (speculative
    verify) can walk back exactly as many tokens as the KV cache drops --
    `length` always equals the state's `composite_len`."""

    state: GrammarState
    length: int
    previous: GrammarTrace | None

    @classmethod
    def start(cls) -> GrammarTrace:
        return cls(state=GrammarState.initial(), length=0, previous=None)

    def advance(self, packed_id: int, ranges: PackedRanges) -> GrammarTrace:
        return GrammarTrace(
            state=self.state.advance(packed_id, ranges), length=self.length + 1, previous=self
        )

    def back(self, n_tokens: int) -> GrammarTrace:
        if n_tokens < 0 or n_tokens > self.length:
            raise ValueError(
                f"cannot step back {n_tokens} tokens from a trace of length {self.length}"
            )
        trace = self
        for _ in range(n_tokens):
            assert trace.previous is not None
            trace = trace.previous
        return trace
