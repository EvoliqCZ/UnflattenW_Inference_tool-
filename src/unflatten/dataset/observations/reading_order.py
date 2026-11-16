"""Per-page PDF reading-order pick (DEC-026 addendum 2, prototype
`pickR v2` — `docs/plans/NALEZ-aligner-dve-nove-triedy-a-navratnost.md`
Doplnok 5).

Poppler's ``-bbox-layout`` (L) reorders words into column-aware reading
order; ``-raw -bbox`` (R) keeps the PDF content-stream order. Neither is
right on every page: L loses a minority of tables and paragraphs to
poppler's column jumps, but switching a whole document to R can move
where a header/footer story is placed and lose it on other pages
(regression class 1, DEC-026 addendum 2).

**The rule, exactly:** align the whole document once under L and once
under R (same parser, :func:`~unflatten.dataset.observations.pdf_text.
extract_pdf_words`). Page ``p`` switches to R iff R matched strictly
more fragments on ``p`` than L did (a tie stays on L). Re-align the
mixed stream (each page's words taken from its chosen order). Then, for
every page that now has fewer matched fragments than it had under L:
un-switch it if it was switched, otherwise un-switch the nearest earlier
switched page (an earlier switch is what changed what this page's
resync sees). Repeat at most ``max_rounds`` times; if a page is still
below its L count when the rounds are exhausted, the whole document
falls back to L. Deterministic — no page ever ends below its L count.

**"Matched on page p" is every fragment on the page, body AND story**
(header/footer/footnote) — not just body words. A body-only count is
blind to regression class 1: a page carrying only a header/footer story
whose count never changes looks unaffected by a switch elsewhere, when
in fact an earlier page's switch moved which page the story places its
supervision on (DEC-026 addendum 2, prototype diag8.py cases
`340fb5c5` p7, `50bb12c5` p4, `8d077bd3` p2, `eeac45f1` p65).

Regression class 2 (a switched page matches more fragments but a
container table fragment lowers its text-coverage *ratio*) is accepted
by the owner and is not guarded here — it is a property of the coverage
ratio, not a bad match.

**T1 x T9c round 4 (2026-09): the PICK DECISION and the RETURNED
ALIGNMENT now use different `align_document` semantics.** Rounds 1-3
(body-only switch decision, node-set loss, bisection rescue) all tried
to make the pick decision itself DEC-033-aware, and each one traded one
real document's regression for another — DEC-033 §3 keep-all section
stories make a page's fragment count a property of the WHOLE document's
story layout, not of that one page's text, and no per-page signal this
module can compute distinguishes that from a genuine loss reliably
enough (measured: `bundle_cc27babb…` never recovered past 3687 of
17969 matched under any of rounds 1-3, despite `bundle_c7022a42…`
recovering cleanly under the same code — the algorithm cannot serve
both shapes at once). The stored evidence settled it: m2a-v7 (pre-DEC-033
code and alignment) converged on `bundle_cc27babb…` with 26 792
observations; only the post-DEC-033 alignment INPUTS regressed the
pick, not the pick algorithm itself.

So the pick decision is now made on **legacy alignment inputs** —
`align_document(..., story_mode="legacy_first_match", frames=False)`,
which reproduces the pre-DEC-033 `align_stories` (first match only, no
keep-all, no section span) and skips the frame pass — using the
UNCHANGED pre-DEC-033 algorithm below (`git show
58bb7fd:src/unflatten/dataset/observations/reading_order.py`, body-only
switch/node-set/bisection all reverted). Once the switched page set (or
fallback) is decided, `pick_reading_order` aligns the CHOSEN stream
exactly once more with the default DEC-033 semantics (keep-all section
stories, frames on) and returns THAT — so the returned alignment still
carries every DEC-033 story placement, `pick_reading_order` just no
longer lets those placements drive which pages switch. A real document
this decouples: the reading-order decision reasons about body text and
first-match stories the same way it did before DEC-033 existed; the
supervision the model actually trains on is still full DEC-033 §3
keep-all.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from unflatten.ast.issues import Issue
from unflatten.dataset.observations.aligner import AlignerPolicy, AlignmentResult, align_document
from unflatten.dataset.observations.pdf_text import PdfPage, PdfWord

_DEFAULT_MAX_ROUNDS = 4


@dataclass(frozen=True)
class ReadingOrderPick:
    """The chosen whole-document alignment plus the evidence Issues the
    pick itself produced (``READING_ORDER_RAW`` per switched page,
    ``READING_ORDER_FALLBACK_LAYOUT`` on whole-document fallback) —
    separate from ``alignment.issues``, which are the aligner's own."""

    alignment: AlignmentResult
    issues: list[Issue]
    switched_pages: list[int]


def _flatten(pages: list[list[PdfWord]]) -> list[PdfWord]:
    return [word for page in pages for word in page]


def _fragments_per_page(alignment: AlignmentResult, n_pages: int) -> list[int]:
    counts = Counter(fragment.page_index for fragment in alignment.fragments)
    return [counts.get(page, 0) for page in range(n_pages)]


def _legacy_align(
    ast: dict, words: list[PdfWord], *, min_key_chars: int | None = None
) -> AlignmentResult:
    """`align_document` under the pre-DEC-033 alignment shape (T1 x T9c
    round 4) — used ONLY to decide which pages switch (module
    docstring)."""
    return align_document(
        ast, words, story_mode="legacy_first_match", frames=False, residual=False,
        min_key_chars=min_key_chars,
    )


def _decide_switched(
    ast: dict,
    layout_words: list[list[PdfWord]],
    raw_words: list[list[PdfWord]],
    n_pages: int,
    max_rounds: int,
) -> list[bool] | None:
    """The pick decision, unchanged from pre-DEC-033
    (`git show 58bb7fd:.../reading_order.py`), run on the legacy
    alignment inputs. Returns the switched-page set, or `None` if the
    rounds exhaust without converging (fall back to L)."""
    layout_alignment = _legacy_align(ast, _flatten(layout_words))
    layout_counts = _fragments_per_page(layout_alignment, n_pages)

    raw_alignment = _legacy_align(ast, _flatten(raw_words))
    raw_counts = _fragments_per_page(raw_alignment, n_pages)

    switched = [raw_counts[p] > layout_counts[p] for p in range(n_pages)]
    if not any(switched):
        return switched

    for _round in range(max_rounds):
        mixed_words = _flatten(
            [raw_words[p] if switched[p] else layout_words[p] for p in range(n_pages)]
        )
        mixed_alignment = _legacy_align(ast, mixed_words)
        mixed_counts = _fragments_per_page(mixed_alignment, n_pages)
        worse_pages = [p for p in range(n_pages) if mixed_counts[p] < layout_counts[p]]
        if not worse_pages:
            return switched
        for p in worse_pages:
            if switched[p]:
                switched[p] = False
                continue
            earlier_switched = [q for q in range(p) if switched[q]]
            if earlier_switched:
                switched[earlier_switched[-1]] = False

    return None


def pick_reading_order(
    ast: dict,
    layout_pages: list[PdfPage],
    raw_pages: list[PdfPage],
    *,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
    policy: AlignerPolicy | None = None,
) -> ReadingOrderPick:
    """Pick a reading order per page and align ``ast`` against it.

    ``layout_pages``/``raw_pages`` are the same PDF extracted under
    :func:`~unflatten.dataset.observations.pdf_text.extract_pdf_words`
    with ``order="layout"`` / ``order="raw"`` respectively. The switch
    decision (`_decide_switched`) runs on legacy (pre-DEC-033) alignment
    inputs; the returned alignment is always built with the default
    (DEC-033) semantics on whichever stream the decision picked (module
    docstring). ``policy`` (`AlignerPolicy`) applies to the returned
    alignment only, never to the decision.
    """
    n_pages = len(layout_pages)
    layout_words = [list(page.words) for page in layout_pages]
    if len(raw_pages) != n_pages:
        # the two pdftotext invocations disagree on page count: nothing
        # to pick between, align under layout order alone
        return ReadingOrderPick(
            alignment=align_document(ast, _flatten(layout_words), policy=policy),
            issues=[],
            switched_pages=[],
        )

    raw_words = [list(page.words) for page in raw_pages]
    switched = _decide_switched(ast, layout_words, raw_words, n_pages, max_rounds)

    if switched is None:
        return ReadingOrderPick(
            alignment=align_document(ast, _flatten(layout_words), policy=policy),
            issues=[
                Issue(
                    code="READING_ORDER_FALLBACK_LAYOUT",
                    severity="warning",
                    node_id=None,
                    path="/",
                    message=(
                        f"reading-order pick did not converge within {max_rounds} rounds; "
                        "whole document falls back to layout order"
                    ),
                )
            ],
            switched_pages=[],
        )

    final_words = _flatten(
        [raw_words[p] if switched[p] else layout_words[p] for p in range(n_pages)]
    )
    final_alignment = align_document(ast, final_words, policy=policy)
    issues = [
        Issue(
            code="READING_ORDER_RAW",
            severity="info",
            node_id=None,
            path=f"/pages/{p}",
            message="page realigned under pdftotext -raw order (DEC-026 addendum 2)",
        )
        for p in range(n_pages)
        if switched[p]
    ]
    return ReadingOrderPick(
        alignment=final_alignment,
        issues=issues,
        switched_pages=[page for page, use_raw in enumerate(switched) if use_raw],
    )
