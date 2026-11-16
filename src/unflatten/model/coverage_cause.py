"""Shared classification of why one page's text coverage falls below a
floor (plan `spec-viacstranove-dokumenty`, tasks P14 and P14 part 2).

**Why this module exists.** `scripts/census/text_coverage_floor.py` (P14)
first split the pages below `--min-text-coverage` into `split_run_denominator_
effect` vs `real_misalignment` (plus two zero/anomaly tail classes) over the
whole corpus. `build_exact_text_cache.text_coverage_summary` (P14 part 2)
needs to report the SAME split in every build's own summary, so the number
stays legible run to run. Reimplementing the arithmetic twice is exactly the
two-scripts-drift risk `text_coverage_floor.py`'s own docstring already names
for its cohort loader (`_load_documents` is loaded dynamically for the same
reason) -- so this module is the one place either caller may compute it.

**What "coverage" means here**, reproduced from `text_authority.py:202-213`:
a page's granted characters (`char_ranges` on THIS page's fragment slice)
against the full length of every text span present on the page, regardless
of how much of that span belongs to this page. A run split across pages is
charged the whole run's length as its denominator on every page it touches;
that is the "split-run denominator effect" this module measures apart from
genuine misalignment.

**The correction.** For a page below the floor, this module computes a
corrected ratio whose denominator is restricted to what could ever be
attributed to THIS page: this page's own granted share, plus whatever
portion of the span was never observed on ANY page (a real gap). Text
legitimately owned by another page of the same document is removed from the
denominator entirely -- never counted against this page. Per span present on
this page: `other_share = total_observed_everywhere - this_page_share`,
`gap = full_length - total_observed_everywhere` (clamped at 0, flagged if
negative -- two pages charged for the same character is a data anomaly, not
a coverage cause), and the corrected contribution is
`this_page_share / (this_page_share + gap)`. When `gap == 0` for every span,
the corrected ratio is exactly 1.0 -- the whole shortfall was boundary
bookkeeping.

**Classification, one page below the floor at a time:**

- `split_run_denominator_effect` -- the corrected ratio clears the floor.
- `real_misalignment` -- the corrected ratio is STILL below the floor: text
  the page should own is not granted to it anywhere in the document.
- `split_run_zero_local_share` -- this page's own share of every present
  span is zero characters, but nothing is missing anywhere (`gap == 0`
  everywhere too) -- the corrected ratio is `0/0`, undefined.
- `empty_span_zero_length` -- same `0/0` corrected ratio, but with no split
  at all: every span present on this page has zero length everywhere.
- `overlap_anomaly` -- a negative `gap` was observed (two pages' `char_ranges`
  claim more of a span than the span contains). Never silently absorbed.

Callers supply `PageNodeShare` (per node present on this page: this page's
own granted characters and the span's full length) rather than the raw
`TextSpanAuthority` objects, because the two callers hold different source
data: `text_coverage_floor.py` has the authorities directly, while
`build_exact_text_cache.build` only carries the row-level summary it already
collected. `PageNodeShare` is the true common denominator between them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: The five classes a sub-floor page can fall into (module docstring).
SUB_FLOOR_CLASSES: tuple[str, ...] = (
    "split_run_denominator_effect",
    "real_misalignment",
    "split_run_zero_local_share",
    "empty_span_zero_length",
    "overlap_anomaly",
)


@dataclass(frozen=True, slots=True)
class PageNodeShare:
    """One text-span node's presence on ONE page: how many characters of it
    THIS page was granted, and the span's full length (the same value on
    every page that carries a slice of it)."""

    node_id: str
    full_length: int
    granted_chars: int


def page_correction(
    page_shares: Sequence[PageNodeShare],
    node_page_chars: Mapping[str, Mapping[Any, int]],
) -> tuple[int, int, bool, bool]:
    """`(corrected_numerator, corrected_denominator, any_split, any_negative_gap)`
    for one page. `node_page_chars` is `node_id -> {page_key: granted_chars_
    on_that_page}`, gathered across EVERY page of the document -- this
    function only reads it, and only needs THIS page's own share (already
    summed into `page_shares`) plus the totals to work out what belongs
    elsewhere."""
    numerator = 0
    denominator = 0
    any_split = False
    any_negative_gap = False
    for share in page_shares:
        total_observed = sum(node_page_chars.get(share.node_id, {}).values())
        other_share = total_observed - share.granted_chars
        gap = share.full_length - total_observed
        if gap < 0:
            any_negative_gap = True
            gap = 0
        if other_share > 0:
            any_split = True
        numerator += share.granted_chars
        denominator += share.granted_chars + gap
    return numerator, denominator, any_split, any_negative_gap


def classify_sub_floor_page(
    page_shares: Sequence[PageNodeShare],
    node_page_chars: Mapping[str, Mapping[Any, int]],
    floor: float,
) -> dict[str, Any]:
    """Classify one page already known to be below `floor` (module
    docstring names the five classes)."""
    numerator, denominator, any_split, any_negative_gap = page_correction(
        page_shares, node_page_chars
    )
    corrected_ratio = None if denominator == 0 else numerator / denominator
    if any_negative_gap:
        cause = "overlap_anomaly"
    elif denominator == 0:
        cause = "split_run_zero_local_share" if any_split else "empty_span_zero_length"
    elif corrected_ratio >= floor:
        cause = "split_run_denominator_effect"
    else:
        cause = "real_misalignment"
    return {
        "cause": cause,
        "corrected_numerator": numerator,
        "corrected_denominator": denominator,
        "corrected_ratio": corrected_ratio,
    }
