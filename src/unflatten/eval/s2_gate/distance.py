"""Edit distance and error rates for the S2 exit gate (RFC005 §37.2, §36.4).

`levenshtein` is a plain iterative two-row DP over any sequence type (str,
tuple, list) — no external edit-distance dependency is added (the
exploration confirmed none is in `pyproject.toml`/`uv.lock`, and adding one
would perturb the environment every identity hash is computed in).

**No normalisation.** RFC005 §37.2: "Normalizacia nesmie odstranit chybu,
ktoru ma Canonical AST zachovat." `character_error_rate` and
`word_error_rate` compare the hypothesis and reference exactly as stored —
no NFC/NFD folding, no case folding, no whitespace collapsing, no
stripping. A precomposed and a decomposed encoding of the same visible
character are therefore different inputs and do not compare as equal.

**Denominator is the reference.** Per RFC005 §36.4, a page with an empty
reference is not a rate of zero (the model could not have "gotten it
right" against nothing) and not a division-by-zero crash: it is a typed
result carrying `rate=None` and a `reason`, so a page like this stays in
the caller's denominator as an error rather than silently vanishing or
silently scoring perfectly. `ErrorRate.rate` is `float | None` precisely
so a caller cannot mistake "no result" for "zero error" — the two are not
the same Python value and cannot be confused by an `if rate:` check either
(`None` and `0.0` are both falsy, but `rate is None` is the only correct
test and the type makes that the natural one to write).

**Word segmentation (WER).** One declared rule: split on Unicode
whitespace via `str.split()` with no arguments, keep everything else as a
token. WER is a secondary diagnostic per §37.2, not the primary text
metric (CER is).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def levenshtein(a: Sequence[object], b: Sequence[object]) -> int:
    """Edit distance between two sequences (insert/delete/substitute, cost 1 each).

    Iterative two-row DP, O(len(a) * len(b)) time, O(min(len(a), len(b)))
    space. Works over any indexable, sized sequence (`str`, `tuple`,
    `list`) since only `len()`, iteration and `==` are used.
    """
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current_row = [i] + [0] * len(b)
        for j, item_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            substitute_cost = previous_row[j - 1] + (0 if item_a == item_b else 1)
            current_row[j] = min(insert_cost, delete_cost, substitute_cost)
        previous_row = current_row
    return previous_row[-1]


@dataclass(frozen=True, slots=True)
class ErrorRate:
    """Result of an error-rate computation, with its denominator (§36.4).

    `rate` is `None`, never `0.0`, when the reference is empty — the
    caller must check `rate is None` before using the value, and `reason`
    names why. When `rate` is a `float`, `reason` is `None`.
    """

    rate: float | None
    distance: int
    reference_length: int
    hypothesis_length: int
    reason: str | None = None


def character_error_rate(hypothesis: str, reference: str) -> ErrorRate:
    """CER = levenshtein(hypothesis, reference) / len(reference).

    Compares raw code points, no normalisation (module docstring). If
    `reference` is empty, returns `rate=None, reason="empty_reference"`
    rather than 0.0 or a `ZeroDivisionError`.
    """
    distance = levenshtein(hypothesis, reference)
    if len(reference) == 0:
        # An empty reference with a non-empty hypothesis has no denominator
        # (all-insertion, undefined rate). Two empty sides is the trivial
        # exact match (distance == 0), which is unambiguously 0.0.
        return ErrorRate(
            rate=0.0 if distance == 0 else None,
            distance=distance,
            reference_length=0,
            hypothesis_length=len(hypothesis),
            reason=None if distance == 0 else "empty_reference",
        )
    return ErrorRate(
        rate=distance / len(reference),
        distance=distance,
        reference_length=len(reference),
        hypothesis_length=len(hypothesis),
    )


def word_error_rate(hypothesis: str, reference: str) -> ErrorRate:
    """WER = levenshtein(hyp_words, ref_words) / len(ref_words).

    Word segmentation rule (stated per module docstring): split on runs of
    Unicode whitespace via `str.split()`, keep everything else as one
    token. Secondary diagnostic per RFC005 §37.2, not the primary metric.
    """
    hyp_words = hypothesis.split()
    ref_words = reference.split()
    distance = levenshtein(hyp_words, ref_words)
    if len(ref_words) == 0:
        return ErrorRate(
            rate=0.0 if distance == 0 else None,
            distance=distance,
            reference_length=0,
            hypothesis_length=len(hyp_words),
            reason=None if distance == 0 else "empty_reference",
        )
    return ErrorRate(
        rate=distance / len(ref_words),
        distance=distance,
        reference_length=len(ref_words),
        hypothesis_length=len(hyp_words),
    )
