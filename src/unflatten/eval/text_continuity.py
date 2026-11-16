"""Per-page text-continuity measurement (RFC005-ERRATA-006 E6 B3).

RFC005-ERRATA-006 §E6 replaces §17.6 with, among others:

    B3 | nulove medzery a duplicity textu | Na kazdom strict-valid vystupe
    strany (preberá odrazku 1 §17.6 v rozsahu strany).

i.e. "zero text gaps or duplicates, on every strict-valid page output, in
page scope." No code measured this before this module: `unflatten.model.
s4_stage_gate_report.gate_b3_no_text_gaps_or_duplicates` was always
`insufficient_evidence` for lack of a producing artefact.

This module compares one page's DECODED text token sequence against its
GOLD text token sequence, deterministically and without a tokenizer:

- **gaps**: contiguous gold spans of `>= min_gap_tokens` tokens that are
  entirely absent from the decode.
- **duplicates**: contiguous decode spans of `>= min_duplicate_tokens`
  tokens that reproduce gold content the decode already emitted elsewhere
  (the decode-loop shape: the same chunk appears more than once).
- **alignment_ratio_ppm**: `difflib.SequenceMatcher.ratio()` between the
  two token sequences, in parts-per-million -- a single scalar continuity
  signal alongside the two span lists, not a substitute for either.

**Text tokens.** Per `unflatten.eval.cohort_gold`'s own rule: packed ids
`>= composite_vocab.STRUCTURAL_COUNT` are text-bearing; everything below
is structural. `decode_text_tokens_for_page` applies the identical filter
to a sealed `PageDecodeResult` so the two sides are comparable.

**Unterminated pages.** A `"max_steps_exceeded"`/`"vocabulary_exhausted"`
page carries no `accepted_tokens` (that field is reserved for a
`"terminated"` page's SEALED hypothesis -- `s2_gate.sealed.
PageDecodeResult`'s own docstring). Its tokens live under
`unterminated_prefix.accepted_tokens` instead (the
`max-steps-discards-the-evidence` fix). This module reads them from
there for exactly those two outcomes, and refuses (raises
`TextContinuityError`) rather than silently treating an empty or missing
prefix as a match.

Deliberately torch-free: only `difflib` (stdlib) and `s2_gate.sealed`'s
plain dataclasses.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass

from unflatten.eval.s2_gate.sealed import PageDecodeResult
from unflatten.model.composite_vocab import STRUCTURAL_COUNT

#: Outcomes that carry decodable tokens at all (module docstring). Every
#: other `PageDecodeResult.outcome` is a pre-decode disposition with
#: nothing to compare.
DECODED_OUTCOMES: frozenset[str] = frozenset(
    {"terminated", "max_steps_exceeded", "vocabulary_exhausted"}
)


class TextContinuityError(ValueError):
    """A page or token sequence cannot be measured for text continuity."""


@dataclass(frozen=True, slots=True)
class TextSpan:
    """A contiguous span, in the token sequence its own detector reports
    positions against (gold positions for a gap, decode positions for a
    duplicate)."""

    start: int
    length: int


@dataclass(frozen=True, slots=True)
class PageTextContinuity:
    """One page's continuity measurement. `has_gap`/`has_duplicate` are
    the two booleans B3 actually cares about; `gaps`/`duplicates` and
    `alignment_ratio_ppm` are the evidence behind them."""

    gold_length: int
    decode_length: int
    gaps: tuple[TextSpan, ...]
    duplicates: tuple[TextSpan, ...]
    alignment_ratio_ppm: int

    @property
    def has_gap(self) -> bool:
        return bool(self.gaps)

    @property
    def has_duplicate(self) -> bool:
        return bool(self.duplicates)


def _text_tokens(raw: Sequence[int]) -> tuple[int, ...]:
    return tuple(t for t in raw if t >= STRUCTURAL_COUNT)


def decode_text_tokens_for_page(page: PageDecodeResult) -> tuple[int, ...]:
    """The page's decoded text token sequence, or raise `TextContinuityError`
    when this page carries nothing to compare -- never an empty tuple
    silently treated as "nothing missing" (module docstring)."""
    if page.outcome not in DECODED_OUTCOMES:
        raise TextContinuityError(
            f"{page.document_bundle_id}: outcome {page.outcome!r} carries no decode tokens "
            "(a pre-decode disposition, not an attempted decode)"
        )
    if page.outcome == "terminated":
        if page.accepted_tokens is None:
            raise TextContinuityError(
                f"{page.document_bundle_id}: outcome is 'terminated' but accepted_tokens is None"
            )
        raw = page.accepted_tokens
    else:
        if page.unterminated_prefix is None:
            raise TextContinuityError(
                f"{page.document_bundle_id}: outcome {page.outcome!r} carries no "
                "unterminated_prefix -- nothing to measure"
            )
        raw = page.unterminated_prefix.accepted_tokens
    tokens = _text_tokens(raw)
    if not tokens:
        raise TextContinuityError(
            f"{page.document_bundle_id}: decode carries zero text tokens after the "
            "structural-token filter -- refused, not counted as a match"
        )
    return tokens


def _contains_contiguous(haystack: tuple[int, ...], needle: tuple[int, ...]) -> bool:
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    return any(haystack[start : start + n] == needle for start in range(len(haystack) - n + 1))


def compute_page_continuity(
    gold_tokens: Sequence[int],
    decode_tokens: Sequence[int],
    *,
    min_gap_tokens: int,
    min_duplicate_tokens: int,
) -> PageTextContinuity:
    """Compare `gold_tokens` against `decode_tokens`. Both sequences must
    be non-empty -- an empty side is refused (`TextContinuityError`), never
    scored as a clean match, per this module's own docstring."""
    if not gold_tokens:
        raise TextContinuityError("gold_tokens is empty -- refused, not a match")
    if not decode_tokens:
        raise TextContinuityError("decode_tokens is empty -- refused, not a match")
    if min_gap_tokens < 1 or min_duplicate_tokens < 1:
        raise TextContinuityError("min_gap_tokens and min_duplicate_tokens must be >= 1")

    gold = tuple(gold_tokens)
    decode = tuple(decode_tokens)
    matcher = difflib.SequenceMatcher(None, gold, decode, autojunk=False)

    gaps: list[TextSpan] = []
    duplicates: list[TextSpan] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            length = i2 - i1
            if length >= min_gap_tokens:
                gaps.append(TextSpan(start=i1, length=length))
        if tag in ("insert", "replace"):
            segment = decode[j1:j2]
            length = j2 - j1
            if length >= min_duplicate_tokens and _contains_contiguous(gold, segment):
                duplicates.append(TextSpan(start=j1, length=length))

    ratio_ppm = round(matcher.ratio() * 1_000_000)
    return PageTextContinuity(
        gold_length=len(gold),
        decode_length=len(decode),
        gaps=tuple(gaps),
        duplicates=tuple(duplicates),
        alignment_ratio_ppm=ratio_ppm,
    )
