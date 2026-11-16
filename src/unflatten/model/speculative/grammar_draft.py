"""Training-free drafting under the composite grammar
(`docs/plans/drafter-research-2026-09-28.md` ideas 2-3, 2026-09-28).

Verification is exact greedy under the grammar mask
(`decode_integration.FreeRunningPrimaryDecoder._verify_block`), so a drafted
token that is illegal after the previous drafted token is certain to be
rejected, and a token that is the ONLY legal id is certain to be accepted
(greedy over a one-element legal set). Both facts are free to exploit:

- :func:`forced_token` -- the single legal id, if there is exactly one;
- :func:`step_if_legal` -- the grammar filter (a draft stops at the first
  illegal token);
- :func:`masked_argmax` -- a head's argmax over the legal ids only (for the
  MTP heads, which otherwise argmax the whole composite vocabulary);
- :class:`GrammarWalkDrafter` -- seed = the primary's own next token, then
  forced tokens and n-gram continuations (`ngram_index`), grammar-filtered;
- :class:`HybridDrafter` -- the n-gram walk when its match is long enough,
  otherwise another drafter (MTP).

A drafter here needs a primary that exposes, besides `next_token`,
`grammar_state_at(prefix)`, `controller` and `vocabulary`
(`FreeRunningPrimaryDecoder` does). Nothing here changes what gets
committed: verification alone decides that.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol

import torch

from unflatten.model.composite_grammar import (
    CompositeGrammarController,
    CompositeGrammarState,
    CompositeGrammarViolation,
)
from unflatten.model.composite_vocab import (
    ByteControlRef,
    CompositeVocabulary,
    StructuralRef,
)
from unflatten.model.exact_text import ByteFrameInvalid
from unflatten.model.free_running_decode import (
    _TEXT_END_LOCAL_ID,
    _legal_packed_ids,
    _packed_text_allowlist,
    _select_greedy,
)
from unflatten.model.speculative.ngram_index import CorpusNgramIndex, PageNgramIndex


def _legal_count(
    controller: CompositeGrammarController, state: CompositeGrammarState,
    vocabulary: CompositeVocabulary,
) -> int:
    """Size of the legal set from `state` without enumerating the ~148k
    text allowlist."""
    domain = controller.allowed(state)
    if domain.mode == "structure":
        return len(domain.structural_local_ids)
    if domain.mode == "text":
        return (
            len(vocabulary.qwen_text_allowlist)
            + int(domain.allow_text_end)
            + int(domain.allow_byte_begin)
        )
    if domain.mode == "byte":
        return sum(hi - lo for lo, hi in domain.allowed_byte_ranges) + int(domain.allow_byte_end)
    return 0


def forced_token(
    controller: CompositeGrammarController, state: CompositeGrammarState,
    vocabulary: CompositeVocabulary,
) -> int | None:
    """The packed id greedy decoding MUST choose from `state` (the legal set
    has exactly one member), or `None`."""
    if _legal_count(controller, state, vocabulary) != 1:
        return None
    return _legal_packed_ids(controller, state, vocabulary)[0]


def step_if_legal(
    controller: CompositeGrammarController, state: CompositeGrammarState,
    vocabulary: CompositeVocabulary, token_id: int,
) -> CompositeGrammarState | None:
    """`controller.step(state, token)` if `token_id` is legal from `state`,
    else `None`."""
    if not 0 <= token_id < vocabulary.ranges.total_count:
        return None
    try:
        return controller.step(state, vocabulary.unpack(token_id))
    except (CompositeGrammarViolation, ByteFrameInvalid):
        return None


@lru_cache(maxsize=8)
def _text_mask(vocabulary: CompositeVocabulary, device: str) -> torch.Tensor:
    """Bool mask over the composite vocabulary: the text allowlist plus
    `TEXT_END` and `byte_begin` -- the whole text-mode legal set, which
    `CompositeGrammarController.allowed` makes identical on every text
    step (both flags are always set there)."""
    mask = torch.zeros(vocabulary.ranges.total_count, dtype=torch.bool)
    mask[list(_packed_text_allowlist(vocabulary))] = True
    mask[vocabulary.pack(StructuralRef(_TEXT_END_LOCAL_ID))] = True
    mask[vocabulary.pack(ByteControlRef("byte_begin"))] = True
    return mask.to(device)


def masked_argmax(
    logits_row: torch.Tensor, controller: CompositeGrammarController,
    state: CompositeGrammarState, vocabulary: CompositeVocabulary,
) -> int | None:
    """Argmax of `logits_row` (`(total_count,)`) over the ids legal from
    `state`, lowest id on ties (`free_running_decode.PINNED_TIE_BREAK`);
    `None` when nothing is legal (terminated)."""
    domain = controller.allowed(state)
    if domain.mode == "text" and domain.allow_text_end and domain.allow_byte_begin:
        mask = _text_mask(vocabulary, str(logits_row.device))
        # `argmax` returns the first maximal index, i.e. the lowest id.
        return int(logits_row.masked_fill(~mask, float("-inf")).argmax())
    legal = _legal_packed_ids(controller, state, vocabulary)
    if not legal:
        return None
    return _select_greedy(logits_row, legal)


class GrammarDraftPrimary(Protocol):
    controller: CompositeGrammarController
    vocabulary: CompositeVocabulary

    def next_token(self, prefix: tuple[int, ...]) -> int | None: ...

    def grammar_state_at(self, prefix: tuple[int, ...]) -> CompositeGrammarState: ...


class GrammarWalkDrafter:
    """Seed = the primary's own next token (always accepted), then up to
    `max_draft` more tokens: a forced token whenever the grammar leaves
    exactly one (`use_forced`), otherwise the longest n-gram continuation
    -- the page's own committed tokens (`page_ngram`) first, then the
    corpus index on a strictly longer match -- stopping at the first
    illegal or missing continuation or at termination. Proposes `()`
    (a plain step, no verify round) when nothing follows the seed.
    `corpus_min_ratio` ignores a corpus continuation seen in less than that
    share of the context's occurrences (count / total).

    `draft_with_match` also returns the match length of the first
    non-forced drafted token (0 if there was none): the hybrid's switch."""

    def __init__(
        self,
        *,
        primary: GrammarDraftPrimary,
        max_draft: int = 7,
        use_forced: bool = True,
        page_ngram: bool = True,
        min_len: int = 2,
        max_len: int = 8,
        corpus: CorpusNgramIndex | None = None,
        corpus_min_ratio: float = 0.0,
    ) -> None:
        if max_draft < 1:
            raise ValueError(f"max_draft must be >= 1, got {max_draft}")
        self._primary = primary
        self._max_draft = max_draft
        self._use_forced = use_forced
        self._page = PageNgramIndex(min_len=min_len, max_len=max_len) if page_ngram else None
        self._corpus = corpus
        self._corpus_min_ratio = corpus_min_ratio

    def _predict(self, context: Sequence[int]) -> tuple[int, int] | None:
        page = self._page.predict(context) if self._page is not None else None
        hit = self._corpus.lookup(context) if self._corpus is not None else None
        if hit is not None and hit[2] >= self._corpus_min_ratio * hit[3]:
            if page is None or hit[1] > page[1]:
                return hit[0], hit[1]
        return page

    def draft_with_match(self, snapshot: Any) -> tuple[tuple[int, ...], int]:
        primary = self._primary
        controller, vocabulary = primary.controller, primary.vocabulary
        committed = tuple(snapshot.committed_tokens)
        seed = primary.next_token(committed)
        if seed is None:
            return (), 0
        if self._page is not None:
            self._page.extend_to(committed)
        state = controller.step(primary.grammar_state_at(committed), vocabulary.unpack(seed))
        context = [*committed, seed]
        drafted: list[int] = []
        match = 0
        while len(drafted) < self._max_draft and state.mode != "terminated":
            token = forced_token(controller, state, vocabulary) if self._use_forced else None
            if token is not None:
                state = controller.step(state, vocabulary.unpack(token))
            else:
                hit = self._predict(context)
                if hit is None:
                    break
                token, length = hit
                nxt = step_if_legal(controller, state, vocabulary, token)
                if nxt is None:
                    break
                if match == 0:
                    match = length
                state = nxt
            drafted.append(token)
            context.append(token)
        return ((seed, *drafted) if drafted else ()), match

    def draft(self, snapshot: Any) -> tuple[int, ...]:
        return self.draft_with_match(snapshot)[0]


class HybridDrafter:
    """The n-gram walk's proposal when its first non-forced token matched a
    context of at least `min_match` tokens, else `fallback.draft` (MTP).
    Counts which one each round used."""

    def __init__(self, *, ngram: GrammarWalkDrafter, fallback: Any, min_match: int) -> None:
        self.ngram = ngram
        self.fallback = fallback
        self.min_match = min_match
        self.ngram_rounds = 0
        self.fallback_rounds = 0

    def draft(self, snapshot: Any) -> tuple[int, ...]:
        tokens, match = self.ngram.draft_with_match(snapshot)
        if tokens and match >= self.min_match:
            self.ngram_rounds += 1
            return tokens
        self.fallback_rounds += 1
        return self.fallback.draft(snapshot)
