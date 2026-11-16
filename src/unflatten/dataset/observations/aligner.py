"""AST ↔ PDF word-stream alignment (RFC003 §24–§25, mapping method
``pdf_text_alignment``).

AST side: per paragraph, the visible text is the NFC concatenation of
its runs, with a tab as ``"\\t"`` and an explicit line break as
``"\\n"`` (mirroring the renderer's text-conservation event derivation
— reimplemented locally, node-id aware). Words are maximal
non-whitespace scalar spans; offsets are half-open ``[start, end)``
over Unicode scalar values of the NFC text.

Matching runs on a PAIRING KEY, not on the raw text (0g): the PDF word
stream is keying-transformed 1:1 in position into
:class:`_StreamWord` (same length and order, so the consumed-bitarray,
the page indices and the table bookkeeping stay aligned with it), and
each AST word's key is computed from its raw word the same way
(``textnorm.normalize_for_matching`` after ``textnorm.apply_caps`` for
the run's ``all_caps``/``small_caps`` spans). The key drops invisible
characters (``Zs``/``Cc``/``Cf``) and expands the ligatures
U+FB00–FB06; cursors advance by key length in both streams, while every
stored offset (``char_start``/``char_end``, fragments, run spans) stays
in the ORIGINAL text's index space — normalization is for pairing only,
the stored text is never rewritten. Words whose key is empty (a
Cf-only word) claim no characters and take no box; the raw stream rides
along for diagnostics (issues show raw text) and for counting
key-assisted matches, which surface as
``OBSERVATION_ALIGNMENT_TEXTNORM_MATCH``.

Alignment is exact sequential word matching (whitespace-insensitive
between words): each paragraph's word sequence must appear verbatim in
the PDF word stream at or after the current position. A paragraph that
cannot be matched yields a typed ``OBSERVATION_ALIGNMENT_MISMATCH``
issue and *no* box — never a guessed one; PDF words no AST node
accounts for (e.g. list marker glyphs rendered from numbering
definitions) yield ``OBSERVATION_ORPHAN_TEXT``.

What that does NOT guarantee — stated here because this module claimed
the opposite for a while, in this file, in the plan and in the tests —
is that a wrong assignment can only surface as a typed mismatch. Exact
character equality cannot tell two cells of identical text apart, so a
stream that lists them in the other order verifies perfectly with the
boxes exchanged. Two checks stand outside the text and catch what it
cannot: :func:`_split_columns` refuses a column cut when two different
assignments both verify, and :func:`_rows_in_document_order` refuses an
assignment whose row bands do not run down the page in the order the
AST states. What holds is the narrower claim: every box is words that
spell their cell, and an assignment contradicting the AST's grid is
refused.

A node spanning multiple pages stays ONE AST node with multiple
fragments: ``fragment_index`` ordering, ``continuation_prev/next``
flags, and node-local char ranges that partition ``[0, len(text))``.
Container nodes (cell, row, table, list_item, list) take the union of
their descendant paragraphs' word boxes; their node-local text is the
``"\\n"``-joined concatenation of descendant paragraph texts.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from unflatten.ast.issues import Issue
from unflatten.dataset.observations.pdf_text import PdfWord
from unflatten.dataset.observations.textnorm import apply_caps, normalize_for_matching
from unflatten.renderer.units import round_half_away_from_zero

if TYPE_CHECKING:
    from unflatten.dataset.observations.stories import StoryAlignment

#: how far ahead of the current stream position a paragraph resyncs by
#: linear scan (skipped PDF words become orphans, e.g. list marker
#: glyphs). Past the window the search continues over the whole
#: remaining stream via :class:`_StreamIndex` — a window-bounded search
#: cannot recover from a block the aligner failed to match (a TOC with
#: dot leaders, a reordered table row), and every later paragraph then
#: stays behind it, which loses the entire document suffix.
_RESYNC_WINDOW = 64

#: how much work one table's peeling may cost before it refuses,
#: counted in words visited: a block charges one per word it holds, and
#: pairing a grid boundary with a corridor charges one per pair. Both
#: are what the run actually does, so the count tracks seconds closely
#: (measured below) while staying a property of the input alone.
#:
#: It is NOT a wall-clock bound, and that is deliberate rather than a
#: shortcut. RFC003's frozen-pixels rule is that the same PDF bytes and
#: the same renderer meta produce a byte-identical sidecar; a timer
#: would make the refusal depend on the machine and the load, so two
#: hosts could derive different `observation_sidecar_hash` values for
#: one document and every `fragment_id` downstream would disagree. This
#: bounds the same thing deterministically.
#:
#: Peeling is a search: every grid boundary a block leaves free can be
#: cut at every corridor of its words, and both halves search again.
#: Distinct blocks are polynomial in the table (:class:`_Search`
#: memoises them); derivations are not, and the first version of this
#: cap counted derivations, which made a span-free 7-column table
#: unresolvable — a regression against the row-at-a-time matcher, on
#: 12.2 % of a 600-document sample of the real corpus. Nothing in this
#: workflow's evidence could see it: the synthetic roster tops out at 4
#: columns (`recipes.py` cols_max 3 and 4), so the 900-document cohort,
#: its byte-identity gate and its 100 % coverage were all silent about
#: any table wider than that. A cohort is evidence about the cohort.
#:
#: Work tracks seconds, measured through `align_document` on span-free
#: grids:
#:
#:     3x8      1 501 units   0.00 s      123x14    475 273   1.48 s
#:     3x20    45 211 units   0.14 s      862x3      21 551   0.13 s
#:     5x40 1 127 701 units   3.56 s      32x40   7 217 281  22.91 s
#:     5x60 5 597 051 units  17.90 s      5x102  46 034 301 148.96 s
#:
#: but not at ONE rate: measured across shapes it spans about 47 000 to
#: 367 000 units per second, the slow end being a block of some 6 000
#: words, where :func:`_corridors` is superlinear in the words while the
#: block is charged one unit per word. So this budget is a ceiling of
#: **3.5 to 6.1 seconds** for one table, measured on refusals, not the
#: ~3 s an average rate suggests. Tightening the charge to match
#: `_corridors` would buy a narrower band and is not done: the point of
#: the bound is that no table runs for 150 seconds, and it holds.
#: Work is about `rows * columns^4 / 11`; over a 600-document sample of
#: the real corpus (1 434 tables) it refuses TWO of them, in two
#: documents (0.3 %): a 19x102 and a 32x40. The 19x102 is the accepted
#: residual against the row-at-a-time matcher, which had no cap; the
#: 32x40 is the price of the ceiling, and 5x102 — which used to succeed
#: after 149 seconds — is what the ceiling is for. A refusal is whole
#: and typed, so the cost is boxes, never wrong ones.
#:
#: Spending it refuses the whole table (:class:`_SearchExhausted`)
#: rather than accepting whatever partial set of candidates happened to
#: run first.
_TABLE_SEARCH_BUDGET = 1_000_000

#: how many times a table group's own AST word count an UNANCHORED
#: bracket (:func:`_bracketed_words` called with ``high=None``, because
#: :func:`_anchor_after` could not locate the group that follows) may
#: take before the region is capped rather than absorbed whole.
#:
#: Without a cap, a bracket that finds no anchor takes every unconsumed
#: word from ``low`` to the end of the document stream — one table's
#: dot-leader-broken lookup then swallows the rest of the document, and
#: every later group tries to match at the end of an already-consumed
#: stream and fails (class A, `NALEZ-aligner-dve-nove-triedy-a-navratnost.md`,
#: measured on the real corpus: 329 documents, 67 222 nodes lost, up to
#: 48 292 of 83 791 stream words taken by one table). The cap keeps the
#: region proportional to what the group could plausibly need instead.
#:
#: OPEN-0b, MEASURED (not 2/3/4 as the design draft proposed): matched
#: text nodes on the four class-A documents named in the design, at
#: k=1/2/3/4 against their release-v11 `stored` count and the design's
#: own floor — cap applies only to the WHOLE-TABLE `_resolve_block`
#: attempt (see below); the per-row band fallback always sees the FULL
#: bracket, uncapped —
#:
#:                stored   k=1   k=2   k=3   k=4   floor
#:     c3126e5f     2185  2065  1920  1867  1779    2000
#:     0f348399     8383  7851  7802  7784  7764    7500
#:     02d3f3ac     4300  4365  4353  4327  4318    4000
#:     ede7a0ce      198   198   188   174   174     198
#:
#: k=1 dominates on every one of the four documents — a TIGHTER cap
#: leaves the resolver less contaminating noise to peel through, not
#: more risk of truncating a real cell — and is the only value that
#: clears every one of the four floors (`ede7a0ce` reaches its floor of
#: 198/198 exactly). It is also the only value that keeps the FULL
#: existing suite green: k=2/3/4 regress
#: `test_align_document_repeated_header_*` and
#: `test_align_document_band_far_past_an_unmatchable_run_still_matches`
#: in tests/dataset/observations/test_aligner_scale_and_repeat.py,
#: hard-won 2026-09-08 fixes for exactly this kind of far, legitimate
#: bracket content — which is why the cap is scoped to the whole-table
#: attempt only, never to the band fallback's own search.
#: The ANCHORED case (`_anchor_after` did locate the next group) is not
#: bounded here — bracketing between two real anchors already limits the
#: region to what lies between them; :data:`_BRACKET_SLACK` is what
#: keeps an UNANCHORED bracket from reaching indefinitely far instead.
_BRACKET_SLACK = 1

#: budget for ONE band's own search inside :func:`_assign_by_bands`
#: (class B, `NALEZ-aligner-dve-nove-triedy-a-navratnost.md`): that
#: function used to share the WHOLE table's :data:`_TABLE_SEARCH_BUDGET`
#: with every band it tries, charging `len(words) - at` per row search
#: (a long table's row count times its word count). A long table
#: exhausts the shared budget after ~135 rows and the whole search
#: aborts (`return assignment`, keeping only the prefix already
#: resolved) — a table of 980 rows then loses every row after the 135th,
#: not because those rows are unresolvable but because the earlier ones
#: spent the budget. A fresh, smaller budget PER BAND means one
#: expensive band can refuse without starving every band after it —
#: the loop already treats "did not resolve" and "budget exhausted" the
#: same way (`missing += 1; continue`, trying the next band with no
#: floor under it), so the exhausted band is reported and skipped,
#: never a reason to give up on the rest of the table.
#:
#: OPEN-0c: the design's own value (200 000) was MEASURED against the
#: two named class-B documents — `3330ae14` (7 739 PDF words) 7 810 >=
#: 7 000; `07234b9b` (29 959 PDF words) 5 524 >= 5 000,
#: `align_document` alone ≈ 5 and ≈ 5.5 minutes respectively on this
#: host. The full sweep (50 000 / 100 000 / 200 000 / 400 000 /
#: 1 000 000) was never completed under the flat `len(words) - at`
#: charge: each candidate costs one full pass over `07234b9b`, and
#: that alone did not finish inside a 590 s window even once. T10 made
#: the row finders charge the work they actually visit, and the
#: per-start rescan that charge then exposed made a single band's scan
#: quadratic in the band's word count — the 22-doc timing sweep at
#: 200 000 finished every document within 1800 s but left 7 documents
#: over 600 s (max_s_overall 1 074.69) — so a tighter 100 000 was
#: carried as a TRIAL value pending the honest sweep. The two-pointer
#: rewrite of the finders below removed the rescan: a band's scan is
#: now one pass over its own words, and the trial failed the class-B
#: floor on the very document it was sized for (`07234b9b` 3 410 <
#: 5 000, where 200 000 yields 5 700). 200 000 is restored: the
#: measured value, and now headroom far above the linear work it
#: prices instead of a guess about a quadratic scan. The honest
#: OPEN-0c-2 sweep is still affordable and will size it properly if
#: the corpus grows.
_BAND_SEARCH_BUDGET = 200_000


#: budget for a table's ENTIRE per-band pass in :func:`_assign_by_bands`,
#: shared by every band it tries (T10). Per-band budgets alone leave the
#: table unbounded: N bands may each spend a full
#: :data:`_BAND_SEARCH_BUDGET` and still refuse, and with the honest
#: per-word charging a refusing band's spend is the work its scan
#: actually did. This is the ceiling the whole band loop spends, per
#: table: one aggregate :class:`_Search` every band's search charges as
#: its parent, so when it runs out the band loop stops for this table
#: and the rows it never reached fall through to :func:`_match_table`'s
#: residual path, named by ``ALIGN_TABLE_BANDS_BUDGET_EXHAUSTED``.
#:
#: PLACEHOLDER, not calibrated: the OPEN-0c-2 sweep sizes it. Sized from
#: the one measurement available before that sweep: the largest class-B
#: document in the current store (`07234b9b`, 29 959 PDF words) spends
#: 28 490 500 words of work across its whole band pass at the former
#: 200 000 per-band budget, and a ceiling at ten full per-band budgets
#: (the first draft of this constant) truncated it after nine bands —
#: 360 of 7 181 nodes matched instead of the 5 700 the pass yields
#: untruncated. 100 000 000 is ~3.5x that measured worst case (1 000
#: full per-band budgets at the 100 000 trial), so no table in the
#: current store truncates while
#: the ceiling still bounds a pathological one. If the sweep finds a
#: larger table, or the per-band budget is lowered (which lowers the
#: honest spend of a refusing band and so this ceiling with it), this
#: constant moves with it.
_TABLE_BANDS_BUDGET = 100_000_000


@dataclass(frozen=True)
class AlignerPolicy:
    """Opt-in alignment rules for a v16 store revision
    (`docs/runs/target-defects-root-cause-2026-09-30.md`). No policy (the
    default everywhere) is today's alignment, byte for byte; a sidecar built
    under a policy records its id (`sidecar.build_observation_sidecar`).

    * `note_marker_tolerance` (A8) -- the residual pass's note-marker forms
      accept a marker LibreOffice paints twice (the automatic number AND
      the `customMarkFollows` text), treat a same-line baseline shift
      (a raised mark) as reading order, and are attempted from
      `note_marker_min_key_chars` keyed characters when the claimed run
      really contains a marker at its seam;
    * `page_break_split` (A9) -- a body paragraph whose words stop matching
      at the end of one page may continue on the next page after the
      page's furniture (footer, header, orphan list labels);
    * `story_field_results` (A10/A11) -- a header/footer paragraph's
      `field` is matched as its computed per-page result (PAGE from the
      section numbering, NUMPAGES from the page count), and a field-only
      header/footer is observed on the one band word that spells it;
    * `story_segment_claims` (A14) -- a header/footer (or frame) paragraph
      no ordered match spells is claimed segment by segment (cut at tabs
      and at fields without a result), in any order, each segment spelled
      exactly once among the page's words no other paragraph of the
      story took (`stories._claim_segments`);
    * `story_frames` (A13) -- a `text_frame` inside a header/footer is
      placed on every page of that story, with field results and segment
      claims, as page-scoped chains -- not once, like a body frame;
    * `repeat_header_rows` (A15) -- a table's leading `repeat_header` rows,
      which Word/LibreOffice repeat on every page the table continues on,
      are observed on each such page by segment claims, as page-scoped
      chains (`stories.align_repeated_header_rows`);
    * `body_tab_segments` (A16) -- a body paragraph no contiguous run
      spells, whose text has tabs, is spelled segment by segment: its
      longest tab segment contiguously, every other segment (a hanging
      label "(2)", a right-tab page number) by the words that spell it on
      the same visual line, beside the neighbouring segment
      (`_span_tab_segments`).
    """

    policy_id: str
    note_marker_tolerance: bool
    note_marker_min_key_chars: int
    page_break_split: bool
    story_field_results: bool
    story_segment_claims: bool
    story_frames: bool
    repeat_header_rows: bool
    body_tab_segments: bool


V16_ALIGNER_POLICY = AlignerPolicy(
    policy_id="v16-align.v1",
    note_marker_tolerance=True,
    note_marker_min_key_chars=8,
    page_break_split=True,
    story_field_results=True,
    story_segment_claims=True,
    story_frames=True,
    repeat_header_rows=True,
    body_tab_segments=True,
)
ALIGNER_POLICIES = {V16_ALIGNER_POLICY.policy_id: V16_ALIGNER_POLICY}


def resolve_aligner_policy(policy_id: str | None) -> AlignerPolicy | None:
    """`None` -> no policy (today's alignment); an unknown id is refused."""
    if policy_id is None:
        return None
    if policy_id not in ALIGNER_POLICIES:
        raise ValueError(
            f"unknown aligner policy {policy_id!r} (known: {sorted(ALIGNER_POLICIES)})"
        )
    return ALIGNER_POLICIES[policy_id]


@dataclass(frozen=True)
class FieldFragment:
    """A `field` node observed on one page by its computed result (policy
    `story_field_results`): the node carries no text, so the observation
    has no char range (sidecar schema 0.5.0 `field_observation_v1`)."""

    node_id: str
    page_index: int
    bbox_ppm: tuple[int, int, int, int]


@dataclass(frozen=True)
class Fragment:
    """One observation fragment of one AST node on one page."""

    node_id: str
    page_index: int
    bbox_ppm: tuple[int, int, int, int]
    char_start: int
    char_end: int
    fragment_index: int
    continuation_prev: bool
    continuation_next: bool


@dataclass
class AlignmentResult:
    fragments: list[Fragment]
    issues: list[Issue]
    matched_text_nodes: int
    total_text_nodes: int
    #: DEC-033 §7 frame coverage counters, folded in from
    #: `frames.align_frames` — see that module for their meaning.
    frames_total: int = 0
    frames_placed: int = 0
    frame_runs_total: int = 0
    frame_runs_observed: int = 0
    frame_text_scalars: int = 0
    #: the residual pass (§3.2/§4.2), folded in only when
    #: `align_document(residual=True)` — see `observations.residual`.
    residual_claimed: int = 0
    residual_refused_ambiguous: int = 0
    residual_refused_contested: int = 0
    residual_not_found: int = 0
    residual_rounds: int = 0
    residual_invariant_skips: int = 0
    note_marker_matches: int = 0
    #: field nodes observed by their result (policy `story_field_results`)
    field_fragments: list[FieldFragment] = field(default_factory=list)


@dataclass
class _MatchedWord:
    """An AST word matched to a PDF word, in node-local char offsets."""

    page_index: int
    bbox_ppm: tuple[int, int, int, int]
    char_start: int
    char_end: int
    #: the match only holds because the pairing key normalized
    #: something (invisible character, ligature, caps) — computed
    #: segment-safely in :func:`_consume`: the raw PDF text under THIS
    #: word differs from the AST word's own raw text, so the same match
    #: would not survive on the raw text at the same tokenization.
    #: Merged PDF tokens (AST ``"foo bar"`` vs one PDF word
    #: ``"foobar"``) do NOT count: the reassembled substring is the AST
    #: word verbatim.
    key_assisted: bool = False
    #: the raw PDF substring under this word, kept only when
    #: ``key_assisted`` — the issue's sample pairs (raw_ast, raw_pdf).
    raw_pdf: str | None = None


@dataclass(frozen=True)
class _StreamWord:
    """One PDF word carrying its pairing key (0g).

    ``text`` is the raw extracted word — kept for diagnostics and for
    reconstructing the raw text under a matched span; ``key`` is the
    normalized form the matching runs in; ``key_to_raw`` maps each key
    character to the raw index it came from (a ligature expansion maps
    every character it produces onto the one raw character).
    """

    page_index: int
    text: str
    key: str
    key_to_raw: tuple[int, ...]
    bbox_ppm: tuple[int, int, int, int]


class AlignmentInvariantViolation(ValueError):
    """A residual claim would give one PDF stream word two AST owners.

    The residual pass (`observations.residual`) only ever searches
    words no paragraph's `spelled_words` already contains, so this can
    only fire on a code defect — see `claim_single_owner`."""


def claim_single_owner(
    node_id: str,
    words: list[_StreamWord],
    owners: dict[int, str],
    *,
    strict: bool,
) -> bool:
    """Register `words` as spelled by `node_id` in `owners` (keyed by
    ``id(word)`` — `_StreamWord` instances are shared, not copied, so
    identity is the word).

    Returns ``True`` and registers every word iff none of them is
    already owned by a *different* node_id. On a conflict, nothing is
    registered: production (`strict=False`) returns ``False`` so the
    caller counts and skips the whole claim (rev 5.1,
    `residual_invariant_skips`); unit tests (`strict=True`) raise
    `AlignmentInvariantViolation` instead, which is how the invariant
    is proven to hold rather than merely assumed.
    """
    conflicts = [w for w in words if owners.get(id(w), node_id) != node_id]
    if conflicts:
        if strict:
            other = owners[id(conflicts[0])]
            raise AlignmentInvariantViolation(
                f"{node_id!r} claims a word already owned by {other!r}"
            )
        return False
    for word in words:
        owners[id(word)] = node_id
    return True


class TableTopologyError(ValueError):
    """A table cell does not state where it sits in its table's grid."""


@dataclass(frozen=True)
class _CellPosition:
    """Where a cell sits in its table's grid — AST topology, which
    ``canonical_ast.schema.json`` makes required on every cell."""

    table_id: str
    row_index: int
    grid_column: int
    col_span: int
    row_span: int


@dataclass(frozen=True)
class _TableGrid:
    """One table's grid: how many columns it has, and which cell owns
    each grid column of each row.

    ``occupancy[row_index][grid_column]`` is the id of the cell covering
    that rectangle: a ``col_span=2`` cell appears under two columns, a
    ``row_span=2`` cell under its column in both rows — including the
    row whose ``cells`` list does not mention it.
    """

    column_count: int
    occupancy: list[dict[int, str]]


@dataclass
class _BlockCell:
    """One cell of a table being assigned words: the grid rectangle it
    covers, which paragraphs of the group it holds, and their AST
    words. ``key`` identifies it inside one table's assignment."""

    key: int
    members: list[int]  # indices into the paragraph group
    row_start: int
    row_end: int
    column_start: int
    column_end: int
    paragraphs: list[list[tuple[str, int, int]]]


class _SearchExhausted(Exception):
    """One table's peeling search hit :data:`_TABLE_SEARCH_BUDGET`.

    Raised rather than returned: a spent budget says nothing about the
    table, so a candidate that came back empty because the search
    stopped must not be read as a candidate that failed to verify.
    Returning None for both let "exactly one assignment verified" mean
    "exactly one was still affordable", which handed a two-cell block
    with ambiguous text a box instead of refusing it.
    """


@dataclass
class _Search:
    """Budget and memo for the blocks of ONE table.

    :func:`_resolve_block` is a pure function of its block, and the same
    block is reached by many derivations — cutting columns 0|1 then 1|2
    lands where cutting 1|2 then 0|1 lands. Without the memo the
    derivations are Catalan in the column count (429 of them at 8
    columns) while the distinct blocks are polynomial, so the memo is
    what keeps an ordinary wide table inside any budget at all.

    ``parent`` is an OUTER budget this search shares its work with:
    :func:`_assign_by_bands` runs every band's own
    :data:`_BAND_SEARCH_BUDGET` search under one per-table
    :data:`_TABLE_BANDS_BUDGET`, and every word of work is charged to
    both. The parent is charged first, so a spend that exhausts BOTH is
    classified as the outer (table-level) event — the coarser of the
    two, and the one whose issue must be emitted only once. ``budget``
    records the size this search was given, so the
    :class:`_SearchExhausted` message states the budget that was
    actually spent, not a constant that happens to be right for one
    call site.
    """

    remaining: int
    memo: dict[Any, dict[int, list[_StreamWord]] | None] = field(default_factory=dict)
    budget: int = 0
    parent: _Search | None = None

    def spend(self, work: int) -> None:
        """Charge ``work`` words visited, or refuse the table."""
        if self.parent is not None:
            self.parent.spend(work)
        self.remaining -= work
        if self.remaining < 0:
            raise _SearchExhausted(
                f"search exceeded its budget of {self.budget} words of work"
            )


@dataclass
class _Paragraph:
    node_id: str
    preorder: int
    path: str
    text: str  # NFC visible text of the paragraph
    run_spans: list[tuple[str, int, int, int]]  # (run_id, preorder, start, end)
    container_ids: tuple[str, ...]  # ancestor containers, outermost first
    row_id: str | None = None  # enclosing table row, if any
    cell_id: str | None = None  # enclosing table cell, if any
    #: innermost enclosing table and the grid position of that table's
    #: cell — all five are None together for a paragraph outside any
    #: table (a nested table's paragraph belongs to the INNER table).
    table_id: str | None = None
    row_index: int | None = None
    grid_column: int | None = None
    col_span: int | None = None
    row_span: int | None = None
    #: paragraph-local offsets where an inline node the AST carries NO
    #: text for renders glyphs anyway — a `field` (its result is
    #: computed at render time and DEC-003 §1.0b keeps it out of the
    #: node) or a `note_ref` (its marker is a property, not run text).
    #: An exact word match across such a point is impossible; the story
    #: pass matches the segments between them instead.
    glyph_gaps: list[int] = field(default_factory=list)
    #: paragraph-local half-open spans of the runs styled `all_caps` /
    #: `small_caps`, collected with the same convention as
    #: `run_spans`: the renderer paints those glyphs uppercase, so the
    #: pairing key uppercases exactly the characters inside (0g).
    caps_spans: list[tuple[int, int]] = field(default_factory=list)
    #: does this paragraph carry a `field` gap (a page number, ...)? The
    #: residual pass (`observations.residual`) never attempts such a
    #: paragraph — its glyphs are per-page and a marker-form retry
    #: would be guessing at a moving target.
    has_field: bool = False
    #: `(offset, marker_text)` for every `note_ref` gap, in the same
    #: paragraph-local offset space as `glyph_gaps` — the residual
    #: pass's note-marker target forms (`observations.residual`) need
    #: the marker text, which the plain keyed words never carry.
    note_refs: list[tuple[int, str]] = field(default_factory=list)
    #: `(offset, field_type, node_id)` for every `field` gap, same offset
    #: space as `glyph_gaps` (policy `story_field_results`)
    field_gaps: list[tuple[int, str, str]] = field(default_factory=list)
    #: `(offset, key)` zero-length entries `_keyed_words` splices in at
    #: their seam -- a field's computed result for the page being tried.
    #: Empty outside a policy attempt, so the keyed words are today's.
    phantoms: list[tuple[int, str]] = field(default_factory=list)
    matched: bool = False
    matched_words: list[_MatchedWord] = field(default_factory=list)
    #: every PDF stream word this paragraph has consumed, appended in
    #: :func:`_consume` alongside `matched_words` and rolled back with
    #: it — the residual pass's "available words" pool (§3.2) is every
    #: stream word no paragraph's `spelled_words` contains.
    spelled_words: list[_StreamWord] = field(default_factory=list)


@dataclass
class _Container:
    node_id: str
    preorder: int
    paragraphs: list[_Paragraph] = field(default_factory=list)


def _topology_field(cell: dict[str, Any], name: str, table_id: str) -> int:
    """One required grid field of ``cell``, or a typed refusal."""
    value = cell.get(name)
    # `True` is 1 in Python, so a bool would pass as a plausible span and
    # produce a wrong grid that every later check agrees with.
    if isinstance(value, bool) or not isinstance(value, int):
        raise TableTopologyError(
            f"cell {cell.get('id')!r} of table {table_id!r} has no integer {name!r}: "
            f"{value!r}. The canonical AST states table topology on every cell; the "
            "aligner maps observed rectangles to owning cells through it and never "
            "infers a grid, because an inferred grid is an inferred box."
        )
    return value


def _table_grid(
    table_id: str, rows: list[dict[str, Any]]
) -> tuple[_TableGrid, list[list[_CellPosition]]]:
    """The table's occupancy grid, plus each cell's position in it.

    A canonical AST states the topology (``grid_column`` / ``col_span``
    / ``row_span`` are required on every cell), so nothing is inferred
    here — not from geometry and not from cell order. A cell that omits
    a field is refused with :class:`TableTopologyError`: this grid maps
    observed rectangles to their owning cells, so an approximated
    occupancy would produce a wrong box that exact character equality
    cannot see: it checks that a cell's words spell the cell, not that
    the grid they were drawn from is the AST's.
    """
    occupancy: list[dict[int, str]] = [{} for _ in rows]
    positions: list[list[_CellPosition]] = []
    for row_index, row in enumerate(rows):
        row_positions: list[_CellPosition] = []
        for cell in row.get("cells", []):
            column = _topology_field(cell, "grid_column", table_id)
            col_span = _topology_field(cell, "col_span", table_id)
            row_span = _topology_field(cell, "row_span", table_id)
            for covered_row in range(row_index, min(row_index + row_span, len(rows))):
                for covered_column in range(column, column + col_span):
                    occupancy[covered_row][covered_column] = cell["id"]
            row_positions.append(
                _CellPosition(
                    table_id=table_id,
                    row_index=row_index,
                    grid_column=column,
                    col_span=col_span,
                    row_span=row_span,
                )
            )
        positions.append(row_positions)
    column_count = max(
        (column + 1 for row_occupancy in occupancy for column in row_occupancy), default=0
    )
    return _TableGrid(column_count=column_count, occupancy=occupancy), positions


class _Walker:
    """Collects paragraphs (with run spans) and container nodes in
    document reading order, assigning preorder indices for stable
    observation ordering."""

    def __init__(self) -> None:
        self.paragraphs: list[_Paragraph] = []
        self.containers: dict[str, _Container] = {}
        self.tables: dict[str, _TableGrid] = {}
        self._counter = 0

    def _next(self) -> int:
        self._counter += 1
        return self._counter

    def walk(self, ast: dict[str, Any]) -> None:
        for section_index, section in enumerate(ast.get("sections", [])):
            for child_index, child in enumerate(section.get("children", [])):
                self._block(child, (), f"/sections/{section_index}/children/{child_index}")

    def walk_blocks(self, blocks: list[dict[str, Any]], path: str) -> None:
        """Collect one story body (header/footer/note children).

        Stories are NOT walked by :meth:`walk`: their glyphs repeat on
        every page of the section, so they are matched per page by
        :mod:`unflatten.dataset.observations.stories`, not by the
        document's single sequential pass.
        """
        for index, block in enumerate(blocks):
            self._block(block, (), f"{path}/children/{index}")

    def _container(self, node: dict[str, Any]) -> _Container:
        container = _Container(node_id=node["id"], preorder=self._next())
        self.containers[node["id"]] = container
        return container

    def _block(
        self,
        node: dict[str, Any],
        ancestors: tuple[str, ...],
        path: str,
        row_id: str | None = None,
        cell_id: str | None = None,
        position: _CellPosition | None = None,
    ) -> None:
        node_type = node.get("type")
        if node_type == "paragraph":
            self._paragraph(node, ancestors, path, row_id, cell_id, position)
        elif node_type == "list":
            self._container(node)
            for i, item in enumerate(node.get("items", [])):
                self._container(item)
                for j, child in enumerate(item.get("children", [])):
                    self._block(
                        child,
                        (*ancestors, node["id"], item["id"]),
                        f"{path}/items/{i}/children/{j}",
                        row_id,
                        cell_id,
                        position,
                    )
        elif node_type == "table":
            self._container(node)
            rows = node.get("rows", [])
            grid, positions = _table_grid(node["id"], rows)
            self.tables[node["id"]] = grid
            for i, row in enumerate(rows):
                self._container(row)
                for j, cell in enumerate(row.get("cells", [])):
                    self._container(cell)
                    for k, child in enumerate(cell.get("children", [])):
                        self._block(
                            child,
                            (*ancestors, node["id"], row["id"], cell["id"]),
                            f"{path}/rows/{i}/cells/{j}/children/{k}",
                            row["id"],
                            cell["id"],
                            positions[i][j],
                        )
        # page_break / column_break / anything else: no glyphs, no walk

    def _paragraph(
        self,
        node: dict[str, Any],
        ancestors: tuple[str, ...],
        path: str,
        row_id: str | None,
        cell_id: str | None,
        position: _CellPosition | None,
    ) -> None:
        preorder = self._next()
        text_parts: list[str] = []
        run_spans: list[tuple[str, int, int, int]] = []
        glyph_gaps: list[int] = []
        caps_spans: list[tuple[int, int]] = []
        has_field = False
        note_refs: list[tuple[int, str]] = []
        field_gaps: list[tuple[int, str, str]] = []
        offset = 0

        def inlines(children: list[dict[str, Any]], offset: int) -> int:
            nonlocal has_field
            for inline in children:
                inline_type = inline.get("type")
                if inline_type == "run":
                    run_preorder = self._next()
                    text = unicodedata.normalize("NFC", inline.get("text", ""))
                    text_parts.append(text)
                    run_spans.append(
                        (inline["id"], run_preorder, offset, offset + len(text))
                    )
                    style = inline.get("style") or {}
                    if style.get("all_caps") or style.get("small_caps"):
                        caps_spans.append((offset, offset + len(text)))
                    offset += len(text)
                elif inline_type == "hyperlink":
                    # a hyperlink is a wrapper, not a break in the text:
                    # its runs are ordinary visible glyphs on the page,
                    # so they belong to this paragraph's word stream
                    # (word-visual-v1; skipping them left 28.7 % of real
                    # documents unmatchable at their first link).
                    self._next()
                    offset = inlines(inline.get("children", []), offset)
                elif inline_type == "tab":
                    text_parts.append("\t")
                    offset += 1
                elif inline_type == "line_break":
                    text_parts.append("\n")
                    offset += 1
                elif inline_type in ("field", "note_ref"):
                    # renders glyphs the AST does not carry (a page
                    # number, a note marker): record the seam, add no text
                    glyph_gaps.append(offset)
                    if inline_type == "field":
                        has_field = True
                        field_gaps.append(
                            (offset, inline.get("field_type", ""), inline.get("id", ""))
                        )
                    else:
                        note_refs.append((offset, inline.get("marker_text", "")))
                # inline_image / floating_image and breaks contribute no
                # matchable glyphs
            return offset

        inlines(node.get("children", []), offset)
        paragraph = _Paragraph(
            node_id=node["id"],
            preorder=preorder,
            path=path,
            text="".join(text_parts),
            run_spans=run_spans,
            glyph_gaps=glyph_gaps,
            caps_spans=caps_spans,
            has_field=has_field,
            note_refs=note_refs,
            field_gaps=field_gaps,
            container_ids=ancestors,
            row_id=row_id,
            cell_id=cell_id,
            table_id=position.table_id if position else None,
            row_index=position.row_index if position else None,
            grid_column=position.grid_column if position else None,
            col_span=position.col_span if position else None,
            row_span=position.row_span if position else None,
        )
        self.paragraphs.append(paragraph)
        for container_id in ancestors:
            self.containers[container_id].paragraphs.append(paragraph)


def _is_glyphless(ch: str) -> bool:
    """A character that renders no glyph: whitespace, or a control/format
    character (``unicodedata`` categories ``Cc``/``Cf``).

    A DEL (U+007F) in a Symbol-family font renders as a zero-width
    bullet — pdftotext emits it as a word with a degenerate box, which
    ``pdf_text.py`` now drops before it reaches this module. Format
    characters (``Cf``) such as ZWJ (U+200D) are invisible joiners, not
    glyphs: they keep matching keys consistent with ``textnorm`` while a
    run consisting only of them stays glyphless, like whitespace-only
    text. Treating them as separators (rather than stripping them) keeps
    AST word offsets in the ORIGINAL text's index space.
    """
    return ch.isspace() or unicodedata.category(ch) in ("Cc", "Cf")


def _words_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Maximal glyph-bearing spans of ``text`` as (word, start, end).

    A separator (whitespace or a control/format character, see
    :func:`_is_glyphless`) is skipped, never stripped: offsets stay in
    the index space of ``text`` as given."""
    words: list[tuple[str, int, int]] = []
    i = 0
    length = len(text)
    while i < length:
        if _is_glyphless(text[i]):
            i += 1
            continue
        j = i
        while j < length and not _is_glyphless(text[j]):
            j += 1
        words.append((text[i:j], i, j))
        i = j
    return words


def _has_glyphs(text: str) -> bool:
    """Does ``text`` carry at least one glyph-bearing word?"""
    return bool(_words_with_offsets(text))


def _union_bbox(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _fragments(node_id: str, words: list[_MatchedWord], total_len: int) -> list[Fragment]:
    """Group ``words`` (node-local offsets, document order) into per-page
    fragments whose char ranges partition ``[0, total_len)``: fragment
    ``k > 0`` starts at its first word's ``char_start``; fragment 0
    starts at 0; every fragment ends where the next starts (the last at
    ``total_len``)."""
    if not words:
        return []
    groups: list[list[_MatchedWord]] = []
    for word in words:
        if groups and groups[-1][0].page_index == word.page_index:
            groups[-1].append(word)
        else:
            groups.append([word])
    starts = [0] + [group[0].char_start for group in groups[1:]]
    ends = starts[1:] + [total_len]
    last = len(groups) - 1
    return [
        Fragment(
            node_id=node_id,
            page_index=group[0].page_index,
            bbox_ppm=_union_bbox([w.bbox_ppm for w in group]),
            char_start=start,
            char_end=end,
            fragment_index=index,
            continuation_prev=index > 0,
            continuation_next=index < last,
        )
        for index, (group, start, end) in enumerate(zip(groups, starts, ends))
    ]


def _orphan_issue(words: list[_StreamWord], path: str) -> Issue:
    return Issue(
        code="OBSERVATION_ORPHAN_TEXT",
        severity="warning",
        node_id=None,
        path=path,
        message=(
            "PDF text-layer words matched no AST node (e.g. renderer-generated "
            "list marker glyphs)."
        ),
        details={
            "words": [w.text for w in words],
            "pages": sorted({w.page_index for w in words}),
            "_word_ids": [id(w) for w in words],
            "_word_pages": [w.page_index for w in words],
        },
    )


def _finalize_orphan_issues(issues: list[Issue], claimed_word_ids: set[int]) -> list[Issue]:
    """Remove residual claims from orphan diagnostics and strip private metadata.

    Body matching emits orphan issues before the residual pass runs. Object
    identities retained privately by :func:`_orphan_issue` let this final step
    remove the exact claimed occurrences, including duplicate text on different
    pages, without changing the published issue shape.
    """
    finalized: list[Issue] = []
    for issue in issues:
        if issue.code != "OBSERVATION_ORPHAN_TEXT":
            finalized.append(issue)
            continue
        word_ids = issue.details.get("_word_ids", ())
        word_pages = issue.details.get("_word_pages", ())
        words = issue.details.get("words", ())
        kept = [
            (word, page)
            for word, page, word_id in zip(words, word_pages, word_ids)
            if word_id not in claimed_word_ids
        ]
        if not kept:
            continue
        finalized.append(
            Issue(
                code=issue.code,
                severity=issue.severity,
                node_id=issue.node_id,
                path=issue.path,
                message=issue.message,
                details={
                    "words": [word for word, _page in kept],
                    "pages": sorted({page for _word, page in kept}),
                },
            )
        )
    return finalized


def _body_section_spans(walker: _Walker) -> dict[int, tuple[int, int]]:
    """Inclusive observed body-page span for each section in ``walker``."""
    spans: dict[int, tuple[int, int]] = {}
    for paragraph in walker.paragraphs:
        parts = paragraph.path.split("/")
        if len(parts) < 3 or parts[1] != "sections":
            continue
        try:
            section_index = int(parts[2])
        except ValueError:
            continue
        for word in paragraph.spelled_words:
            lo, hi = spans.get(section_index, (word.page_index, word.page_index))
            spans[section_index] = (min(lo, word.page_index), max(hi, word.page_index))
    return spans


def _section_margin_bands(section: dict[str, Any]) -> list[tuple[int, int]]:
    """Top and bottom page-margin bands in normalized page ppm."""
    height_mpt = section.get("page", {}).get("height_mpt")
    margins = section.get("margins", {})
    if not isinstance(height_mpt, int) or height_mpt <= 0:
        return []

    def extent_ppm(value: Any) -> int:
        if not isinstance(value, int) or value <= 0:
            return 0
        ppm = round_half_away_from_zero(value * 1_000_000, height_mpt)
        return min(max(ppm, 0), 1_000_000)

    top = extent_ppm(margins.get("top_mpt"))
    bottom = extent_ppm(margins.get("bottom_mpt"))
    bands: list[tuple[int, int]] = []
    if top:
        bands.append((0, top))
    if bottom:
        bands.append((1_000_000 - bottom, 1_000_000))
    return bands


def _margins_by_page(
    ast: dict[str, Any], section_spans: dict[int, tuple[int, int]]
) -> dict[int, list[tuple[int, int]]]:
    """Map each section page to its top and bottom margin bands."""
    sections = ast.get("sections", [])
    result: dict[int, list[tuple[int, int]]] = {}
    for section_index, (first_page, last_page) in section_spans.items():
        if not (0 <= section_index < len(sections)):
            continue
        bands = _section_margin_bands(sections[section_index])
        if not bands:
            continue
        for page_index in range(first_page, last_page + 1):
            result.setdefault(page_index, []).extend(bands)
    return result


def _mismatch_issue(paragraph: _Paragraph, texts: list[str], stream: list[str]) -> Issue:
    return Issue(
        code="OBSERVATION_ALIGNMENT_MISMATCH",
        severity="error",
        node_id=paragraph.node_id,
        path=paragraph.path,
        message="paragraph text not found in the PDF text-layer word stream.",
        details={"expected_words": texts[:16], "stream_words": stream[:16]},
    )


def _assembled_raw(
    covering: list[tuple[int, int, _StreamWord]], start: int, end: int
) -> str:
    """The raw PDF text under a key range, in stream order.

    Each covering word contributes the raw substring spanning the raw
    characters its overlapping key characters came from — a ligature
    therefore yields its single raw glyph, and an invisible character
    between two kept ones is included: it is on the page."""
    parts: list[str] = []
    for key_start, _key_end, word in covering:
        lo = max(0, start - key_start)
        hi = min(len(word.key), end - key_start)
        if lo >= hi:
            continue
        raws = [word.key_to_raw[k] for k in range(lo, hi)]
        parts.append(word.text[min(raws) : max(raws) + 1])
    return "".join(parts)


def _page_groups(
    covering: list[tuple[int, int, _StreamWord]],
) -> list[list[tuple[int, int, _StreamWord]]]:
    groups: list[list[tuple[int, int, _StreamWord]]] = []
    for span in covering:
        if groups and groups[-1][-1][2].page_index == span[2].page_index:
            groups[-1].append(span)
        else:
            groups.append([span])
    return groups


def _raw_piece_lengths(raw_len: int, key_lengths: list[int]) -> list[int]:
    count = len(key_lengths)
    if count == 1:
        return [raw_len]
    total_key = sum(key_lengths)
    if total_key == 0:
        base, extra = divmod(raw_len, count)
        lengths = [base + (1 if i < extra else 0) for i in range(count)]
    else:
        exact = [raw_len * length / total_key for length in key_lengths]
        lengths = [int(value) for value in exact]
        remainder = raw_len - sum(lengths)
        if remainder:
            order = sorted(
                range(count),
                key=lambda i: (exact[i] - lengths[i], key_lengths[i]),
                reverse=True,
            )
            for index in order:
                if remainder == 0:
                    break
                lengths[index] += 1
                remainder -= 1
    if raw_len >= count:
        for index, length in enumerate(lengths):
            if length:
                continue
            donor = max(
                (j for j in range(count) if j != index),
                key=lambda j: lengths[j],
            )
            lengths[index] = 1
            lengths[donor] -= 1
    return lengths


def _consume(paragraph: _Paragraph, ast_words: list[tuple[str, int, int]],
             pdf_words: list[_StreamWord],
             *,
             global_owned_ids: set[int] | None = None) -> None:
    """Attach boxes to AST words by overlapping their character spans.

    AST word boundaries come from whitespace; PDF word boundaries come
    from position on the page. A superscript footnote marker turns one
    AST word into three PDF words, so a 1:1 zip loses the paragraph. The
    two streams agree on *characters*, not on tokens, so each AST word
    takes the union of the PDF words whose characters overlap it.

    Runs on the pairing key (0g): ``ast_words`` is
    ``(key, char_start, char_end)`` with the offsets still in the
    ORIGINAL raw text, and ``pdf_words`` carry their key. A word whose
    reassembled raw PDF text differs from the AST word's own raw text is
    flagged ``key_assisted`` — the match held only because the key
    normalized something.
    """
    if global_owned_ids is not None:
        if any(id(word) in global_owned_ids for word in pdf_words):
            return
        for word in pdf_words:
            global_owned_ids.add(id(word))
    paragraph.matched = True
    paragraph.spelled_words.extend(pdf_words)
    # concatenated key spans, whitespace removed, in both streams
    pdf_spans: list[tuple[int, int, _StreamWord]] = []
    cursor = 0
    for word in pdf_words:
        pdf_spans.append((cursor, cursor + len(word.key), word))
        cursor += len(word.key)

    cursor = 0
    for key, char_start, char_end in ast_words:
        start, end = cursor, cursor + len(key)
        cursor = end
        covering = [span for span in pdf_spans if span[0] < end and span[1] > start]
        if not covering:
            continue
        groups = _page_groups(covering)
        if len(groups) == 1:
            boxes = [word.bbox_ppm for _a, _b, word in covering]
            raw_ast = paragraph.text[char_start:char_end]
            raw_pdf = _assembled_raw(covering, start, end)
            key_assisted = raw_ast != raw_pdf
            paragraph.matched_words.append(
                _MatchedWord(
                    page_index=covering[0][2].page_index,
                    bbox_ppm=_union_bbox(boxes),
                    char_start=char_start,
                    char_end=char_end,
                    key_assisted=key_assisted,
                    raw_pdf=raw_pdf if key_assisted else None,
                )
            )
            continue
        key_lengths = [
            sum(
                max(0, min(key_end, end) - max(key_start, start))
                for key_start, key_end, _word in group
            )
            for group in groups
        ]
        raw_len = char_end - char_start
        if not any(key_lengths) or raw_len <= 0:
            boxes = [word.bbox_ppm for _a, _b, word in covering]
            raw_ast = paragraph.text[char_start:char_end]
            raw_pdf = _assembled_raw(covering, start, end)
            key_assisted = raw_ast != raw_pdf
            paragraph.matched_words.append(
                _MatchedWord(
                    page_index=covering[0][2].page_index,
                    bbox_ppm=_union_bbox(boxes),
                    char_start=char_start,
                    char_end=char_end,
                    key_assisted=key_assisted,
                    raw_pdf=raw_pdf if key_assisted else None,
                )
            )
            continue
        lengths = _raw_piece_lengths(raw_len, key_lengths)
        position = char_start
        for group, length in zip(groups, lengths):
            piece_start = position
            piece_end = position + length
            position = piece_end
            boxes = [word.bbox_ppm for _a, _b, word in group]
            raw_piece = paragraph.text[piece_start:piece_end]
            raw_pdf = _assembled_raw(group, start, end)
            key_assisted = raw_piece != raw_pdf
            paragraph.matched_words.append(
                _MatchedWord(
                    page_index=group[0][2].page_index,
                    bbox_ppm=_union_bbox(boxes),
                    char_start=piece_start,
                    char_end=piece_end,
                    key_assisted=key_assisted,
                    raw_pdf=raw_pdf if key_assisted else None,
                )
            )


class _StreamIndex:
    """Positions of each distinct PDF word text, for resynchronising
    beyond :data:`_RESYNC_WINDOW` without scanning the stream.

    A run of PDF words can only spell an AST paragraph if its first word
    is a prefix of that paragraph's characters, so the candidate starts
    are the positions of the paragraph's own character prefixes — at
    most ``max_word_len`` dictionary lookups instead of a full scan."""

    def __init__(self, pdf_words: list[_StreamWord]) -> None:
        self.positions: dict[str, list[int]] = {}
        self.max_word_len = 0
        for index, word in enumerate(pdf_words):
            if not word.key:
                continue
            self.positions.setdefault(word.key, []).append(index)
            self.max_word_len = max(self.max_word_len, len(word.key))

    def candidate_starts(self, target: str, pos: int) -> list[int]:
        starts: set[int] = set()
        for length in range(1, min(len(target), self.max_word_len) + 1):
            for index in self.positions.get(target[:length], ()):
                if index >= pos:
                    starts.add(index)
        return sorted(starts)


def _span_at(target: str, pdf_words: list[_StreamWord], start: int) -> tuple[int, int] | None:
    """Does a run of PDF words starting at ``start`` spell ``target``?"""
    accumulated = ""
    end = start
    while end < len(pdf_words) and len(accumulated) < len(target):
        accumulated += pdf_words[end].key
        end += 1
        if accumulated == target:
            return (start, end)
        if not target.startswith(accumulated):
            return None
    return None


_NEXT_GROUP_LOCATE_WINDOW = 4 * _RESYNC_WINDOW
_OPENING_WORDS = 3


def _opening_span(
    prefix: list[str], pdf_words: list[_StreamWord], start: int
) -> tuple[int, int] | None:
    """Where the next group's opening words begin, in order.

    The words may be interleaved with list markers or page furniture,
    so this is a subsequence check inside ``_RESYNC_WINDOW`` rather than
    one contiguous character span. It returns the span of the first
    word, not of the whole prefix, because the caller needs the stream
    index where the following group starts.
    """
    cursor = start
    limit = min(len(pdf_words), start + _RESYNC_WINDOW + 1)
    first: int | None = None
    for key in prefix:
        span = None
        for index in range(cursor, limit):
            span = _span_at(key, pdf_words, index)
            if span is not None:
                break
        if span is None:
            return None
        if first is None:
            first = span[0]
        cursor = span[1]
    assert first is not None
    return (first, cursor)


def _locate_next_group_opening(
    next_texts: list[str], pdf_words: list[_StreamWord], pos: int
) -> int | None:
    """Stream index where the following group begins, or None.

    Bounded to ``_NEXT_GROUP_LOCATE_WINDOW``: this fallback exists for a
    heading that sits between the current position and its table, with
    page furniture in between — not for locating a group on a later
    page and resynchronising across the document.
    """
    prefix = [key for key in next_texts[:_OPENING_WORDS] if key]
    if not prefix:
        return None
    limit = min(len(pdf_words), pos + _NEXT_GROUP_LOCATE_WINDOW + 1)
    for start in range(pos, limit):
        span = _opening_span(prefix, pdf_words, start)
        if span is not None:
            return span[0]
    return None


def _find_char_span(
    texts: list[str],
    pdf_words: list[_StreamWord],
    pos: int,
    index: _StreamIndex | None = None,
    corroboration: list[str] | None = None,
) -> tuple[int, int] | None:
    """Run of PDF words whose key concatenation equals the AST keys'.

    Matches on the character stream rather than on tokens: the two agree
    on characters, but a superscript or any repositioned run splits an
    AST word into several PDF words (`zm.1))` -> `zm.` `1)` `)`), which
    token matching cannot survive. ``texts`` are the AST words' pairing
    keys (0g); the match is a property of the keys, and the returned
    span indexes the same stream the caller keyed.

    Searches ``_RESYNC_WINDOW`` words ahead of ``pos`` first — the
    nearest occurrence is the right one — and, given ``index``, over the
    whole remaining stream after that, so an unmatched block upstream
    does not hide every later paragraph. ``corroboration`` is the next
    group's words: a far resync is accepted only if they follow the
    span, which keeps a heading that also appears in a table of
    contents from dragging the position past the body in between.
    """
    target = "".join(texts)
    if not target:
        return (pos, pos)
    limit = min(len(pdf_words), pos + _RESYNC_WINDOW + 1)
    for start in range(pos, limit):
        span = _span_at(target, pdf_words, start)
        if span is not None:
            return span
    if index is None:
        return None
    # Beyond the window the nearest occurrence is no longer evidence
    # that it is the right one, so resync only on an unambiguous
    # target: two candidate occurrences mean the paragraph cannot be
    # placed, which is a typed mismatch, not a guess.
    found: tuple[int, int] | None = None
    for start in index.candidate_starts(target, limit):
        span = _span_at(target, pdf_words, start)
        if span is None:
            continue
        if found is not None:
            return None
        found = span
    if found is None or corroboration is None:
        return found
    if _find_char_span(corroboration, pdf_words, found[1]) is None:
        return None
    return found


def _find_row_segment(
    target: str,
    pdf_words: list[_StreamWord],
    pos: int,
    window: int | None = _RESYNC_WINDOW,
    search: _Search | None = None,
) -> tuple[int, int] | None:
    """Run of PDF words holding exactly the row's characters.

    A row cannot be located with ``_find_char_span``: the stream
    interleaves wrapped cell lines across the row, so the row's
    characters appear in a different *order* than in the AST — only the
    multiset is shared. Token counts are not shared either (a split or
    joined word changes them), so the segment is bounded by characters.

    ``window`` bounds how far past ``pos`` the search starts, because
    the nearest occurrence is the right one when the position is
    trustworthy. :func:`_assign_by_bands` passes None: there the
    position is the end of the last band that resolved, every band
    before an unresolvable one leaves it behind, and the search runs
    inside one table's own words, where a band cannot be confused with
    anything but another band of the same table. Measured on the real
    corpus, the window cost one document 108 of the 124 nodes it can
    have and another 150.

    One two-pointer pass, not a rescan per start (T10): as ``start``
    advances, the window ``[start, end)`` keeps its right edge — every
    word enters the window at most once, so the search is linear in
    the words it may reach. The per-start fresh ``Counter`` it
    replaced re-read the stream once per start position; a band that
    resolves near the end of a long run of near-miss words paid
    starts × words, which is what the honest charging now prices.

    ``search``, when given, is charged one word of work per word the
    window gains — the honest cost of the scan, in the same units
    :func:`_resolve_block` pays. A caller that pre-charges a flat
    window instead of passing ``search`` prices words never visited and
    leaves the worst-case scan free, which is the T10 defect; the band
    loop is the only caller that prices, and it prices itself.
    """
    if not target:
        return (pos, pos)
    expected = Counter(target)
    need = len(target)
    limit = len(pdf_words) if window is None else min(len(pdf_words), pos + window + 1)
    counts: Counter[str] = Counter()
    length = 0
    over = 0
    start = pos
    end = pos
    while start < limit:
        if end < start:
            end = start
        while end < len(pdf_words) and length < need:
            key = pdf_words[end].key
            length += len(key)
            for char, n in Counter(key).items():
                before = counts[char]
                after = before + n
                over += max(0, after - expected[char]) - max(0, before - expected[char])
                counts[char] = after
            if search is not None:
                search.spend(1)
            end += 1
        if length == need and over == 0:
            return (start, end)
        if start < end:
            key = pdf_words[start].key
            length -= len(key)
            for char, n in Counter(key).items():
                before = counts[char]
                after = before - n
                over += max(0, after - expected[char]) - max(0, before - expected[char])
                counts[char] = after
                if after == 0:
                    del counts[char]
        elif end == len(pdf_words):
            break
        start += 1
        if length < need and end == len(pdf_words):
            break
    return None


#: how many characters neither in a row's own text nor already needed by
#: it (0f, "tolfb": `NALEZ-aligner-dve-nove-triedy-a-navratnost.md`
#: "Doplnok 4") the tolerant row fallback below may accept inside its
#: window before refusing: a merged neighbouring cell's text interleaved
#: with the row's own, or the row's own characters repeated once too
#: often. `_find_row_segment` refuses BOTH — any foreign character or
#: any character in excess is enough to break the exact multiset match —
#: which is right for locating a row precisely but loses rows real
#: tables genuinely have overlapping content around. za-tables prototype
#: `cm_tolfb` (~/unflatten-runs/za-tables/proto.py), value UNCALIBRATED
#: here — kept at the prototype's own 32.
_TOLERANT_ROW_FOREIGN_BUDGET = 32


def _tolerant_row_segment(
    target: str,
    pdf_words: list[_StreamWord],
    pos: int,
    window: int | None = _RESYNC_WINDOW,
    search: _Search | None = None,
) -> tuple[int, int] | None:
    """Like :func:`_find_row_segment`, but tolerates up to
    :data:`_TOLERANT_ROW_FOREIGN_BUDGET` characters neither in ``target``
    nor already accounted for by it, within the window.

    A fallback ONLY, tried by :func:`_assign_by_bands` after
    :func:`_find_row_segment` fails to place a band at all — never for
    the "is this the only place it could be" uniqueness recheck the
    caller runs afterwards, which stays on the exact finder, and never
    on its own result without the caller's guard against swallowing the
    row right after it (see :func:`_tolerant_window_hides_a_later_row`):
    measured on the real corpus, an earlier version of this fallback
    with no such guard lost 461 nodes across 5 documents — one tolerant
    match ate the next row's own words as "foreign" noise and pushed the
    search past it.

    The trailing edge is exact — the scan stops the word after every
    character of ``target`` is accounted for, tolerated ones included —
    but the LEADING edge can start on a run of purely-foreign words
    before ``target``'s own characters begin, so those are trimmed back
    off before returning, the same as :func:`_find_row_segment` never
    needs to (its start is already exact). The budget is a ceiling on
    the TRIMMED span, not a stop condition the completing word may
    silently cross: a leading word is trimmable only while removing it
    still leaves every target character satisfied, and the match is
    refused if the trimmed span's unneeded characters exceed the budget.

    The scan is one two-pointer pass like :func:`_find_row_segment`'s
    (T10): as ``start`` advances the window keeps its right edge, so
    each word is counted at most once. ``search``, when given, is
    charged one word of work per word the window gains and per leading
    word the trim eats — the same honest accounting
    :func:`_find_row_segment` pays.
    """
    if not target:
        return (pos, pos)
    expected = Counter(target)
    need = len(target)
    limit = len(pdf_words) if window is None else min(len(pdf_words), pos + window + 1)
    counts: Counter[str] = Counter()
    length = 0
    matched = 0
    start = pos
    end = pos
    while start < limit:
        if end < start:
            end = start
        while end < len(pdf_words) and matched < need:
            key = pdf_words[end].key
            length += len(key)
            for char, n in Counter(key).items():
                before = counts[char]
                after = before + n
                matched += min(after, expected[char]) - min(before, expected[char])
                counts[char] = after
            if search is not None:
                search.spend(1)
            end += 1
        if matched == need:
            while start < end - 1:
                key = pdf_words[start].key
                word_counts = Counter(key)
                if all(
                    counts[char] - word_counts[char] >= expected[char]
                    for char in word_counts
                ):
                    length -= len(key)
                    for char, n in word_counts.items():
                        before = counts[char]
                        after = before - n
                        matched += min(after, expected[char]) - min(
                            before, expected[char]
                        )
                        counts[char] = after
                        if after == 0:
                            del counts[char]
                    if search is not None:
                        search.spend(1)
                    start += 1
                else:
                    break
            if matched == need and length - need <= _TOLERANT_ROW_FOREIGN_BUDGET:
                return (start, end)
        if start < end:
            key = pdf_words[start].key
            length -= len(key)
            for char, n in Counter(key).items():
                before = counts[char]
                after = before - n
                matched += min(after, expected[char]) - min(before, expected[char])
                counts[char] = after
                if after == 0:
                    del counts[char]
        elif end == len(pdf_words):
            break
        start += 1
        if matched < need and end == len(pdf_words):
            break
    return None


def _tolerant_window_hides_a_later_row(
    remaining: list[_BlockCell], words: list[_StreamWord], at: int, window_end: int
) -> bool:
    """Would a tolerant match spanning ``words[at:window_end]`` already
    contain, exactly, the next still-unplaced row's own text?

    If so the match just accepted is not this band's glyphs tolerating
    some noise — it is the NEXT row's glyphs, tolerated as the noise —
    and accepting it would advance the cursor past that row and lose it
    (the 461-node regression :func:`_tolerant_row_segment` documents).
    Checking only the row immediately after is deliberate: it is the one
    whose search starts exactly at the cursor this match would set, so
    it is the one a swallow costs first, and it is cheap to name here
    without re-running the whole band loop to find out.
    """
    if not remaining:
        return False
    next_row = min(cell.row_start for cell in remaining)
    next_cells = [cell for cell in remaining if cell.row_start == next_row]
    target = "".join(word[0] for cell in next_cells for p in cell.paragraphs for word in p)
    if not target:
        return False
    return any(_span_at(target, words, start) is not None for start in range(at, window_end))


def _bracketed_words(
    pdf_words: list[_StreamWord],
    consumed: bytearray,
    low: _StreamWord | None,
    high: _StreamWord | None,
) -> list[int]:
    """Stream indices of the words that lie below ``low`` and above
    ``high`` — a block's glyphs located by where they are rather than by
    where they are read.

    A `col_span` cell can erase a table's middle gutter, and poppler
    then reads the whole PAGE as two column flows and defers the
    right-hand one to the end of the stream: in document 17 of
    `wcv0-tables` the two tables' stream ranges overlap ([5, 96] and
    [54, 100]), so no contiguous run of the stream is either table.
    Vertically they do not overlap at all, and the paragraphs around
    them are matched, so the anchors bracket each table exactly: the
    region's PDF word count equals the table's own AST word count for
    100 tables of 100, documents 17 and 43 included.

    The cost of the bracket is that it presumes nothing but the block
    sits between its anchors. Where that is false the region absorbs
    foreign words, and a cell whose text they do not spell can only
    become an orphan or a typed mismatch. They CAN become a wrong box
    where they spell it — a stray glyph repeating a cell's own text is
    exactly that, and the claim that a wrong assignment could only
    surface as a mismatch was false; see
    :func:`_rows_in_document_order` for the half of it that is now
    checked, and the module docstring for what is left.

    UNPROVEN individually: ``low`` and the ``consumed`` skip are two
    ways of saying the same thing here, and nothing distinguishes
    either one alone — dropping ``low``, softening it to the anchor's
    top, or dropping the skip each leaves the whole suite green and the
    900-document cohort byte-identical (mutated all three ways,
    2026-09-08); dropping BOTH fails. They are not equivalent in
    general — the skip also excludes a matched word above the anchor,
    ``low`` also excludes an unmatched one — but no document here shows
    it. Do not read the green suite as evidence that either catches
    anything.
    """
    start = (low.page_index, low.bbox_ppm[3]) if low is not None else None
    end = (high.page_index, high.bbox_ppm[1]) if high is not None else None
    picked: list[int] = []
    for index, word in enumerate(pdf_words):
        if consumed[index]:
            continue
        if start is not None and (word.page_index, word.bbox_ppm[1]) < start:
            continue
        if end is not None and (word.page_index, word.bbox_ppm[3]) > end:
            continue
        picked.append(index)
    return picked


def _corridors(words: list[_StreamWord]) -> list[tuple[int, int]]:
    """Vertical strips of the x-axis that no word of ``words`` occupies.

    The words' x-intervals merge where they overlap; what is left
    between the merged bands is empty across *every* word in the cloud.
    Given a table row's word cloud, those strips are the only places a
    column boundary can be — and all of them are offered, because
    :func:`_cell_clusters` chooses between them by exact character
    equality. The x-band clustering this replaced instead ranked the
    same gaps by width, which reads document 7 of `wcv0-tables`
    backwards: its inter-cell gutter is 2.8 pt and its intra-cell word
    space 3.0 pt, so the wrong one won and a word crossed into the next
    cell.

    Threshold-free by construction: emptiness is not a width, and
    nothing here prefers one corridor over another.
    """
    bands: list[tuple[int, int]] = []
    for word in words:
        x1, _, x2, _ = word.bbox_ppm
        overlapping = [b for b in bands if x1 < b[1] and x2 > b[0]]
        merged = (
            min([x1] + [b[0] for b in overlapping]),
            max([x2] + [b[1] for b in overlapping]),
        )
        bands = [b for b in bands if b not in overlapping] + [merged]
    bands.sort()
    return [(bands[i][1], bands[i + 1][0]) for i in range(len(bands) - 1)]


def _cell_spans(
    cluster: list[_StreamWord], paragraphs: list[list[tuple[str, int, int]]]
) -> list[tuple[int, int]] | None:
    """Where each paragraph of one cell sits inside ``cluster``, or None
    if the cluster does not spell them in order.

    The comparison is :func:`_find_char_span` — the same exact
    character equality every other branch uses — so this is a
    verification of a proposed cell, not a similarity score.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for ast_words in paragraphs:
        span = _find_char_span([w[0] for w in ast_words], cluster, cursor)
        if span is None:
            return None
        spans.append(span)
        cursor = span[1]
    return spans


def _table_cells(
    group: list[_Paragraph], words_per_paragraph: list[list[tuple[str, int, int]]]
) -> list[_BlockCell]:
    """The table's cells that carry words, in AST row-major order."""
    cells: list[_BlockCell] = []
    for index, paragraph in enumerate(group):
        if cells and group[cells[-1].members[0]].cell_id == paragraph.cell_id:
            cells[-1].members.append(index)
            cells[-1].paragraphs.append(words_per_paragraph[index])
            continue
        # A group of more than one paragraph is one table by
        # construction (:func:`_match_paragraphs` groups on a non-None
        # ``table_id``), so the walker has filled every grid field. This
        # is a refusal rather than an ``assert`` because it is a
        # statement about the AST that reached us, and `python -O` drops
        # asserts: an unchecked None here would reach the arithmetic
        # below and produce a grid, which is a box.
        if (
            paragraph.row_index is None
            or paragraph.grid_column is None
            or paragraph.row_span is None
            or paragraph.col_span is None
        ):
            raise TableTopologyError(
                f"paragraph {paragraph.node_id!r} is grouped as table "
                f"{paragraph.table_id!r} but states no grid position"
            )
        cells.append(
            _BlockCell(
                key=len(cells),
                members=[index],
                row_start=paragraph.row_index,
                row_end=paragraph.row_index + paragraph.row_span,
                column_start=paragraph.grid_column,
                column_end=paragraph.grid_column + paragraph.col_span,
                paragraphs=[words_per_paragraph[index]],
            )
        )
    return [cell for cell in cells if any(cell.paragraphs)]


def _x_clusters_stream(
    words: list[_StreamWord], cluster_count: int
) -> list[list[_StreamWord]] | None:
    """Cluster words into exactly ``cluster_count`` horizontal bands by
    x-position (one band per table cell of a row, left to right).

    Overlapping x-intervals merge into bands (wrapped lines of one cell
    overlap vertically-stacked words), then bands are merged across the
    *smallest* horizontal gaps until ``cluster_count`` remain.  Verified
    afterwards by :func:`_cell_spans`, so a wrong merge yields a typed
    mismatch, never a silently wrong box.
    """
    spans: list[tuple[int, int]] = []
    for word in words:
        x1, _, x2, _ = word.bbox_ppm
        overlapping = [s for s in spans if x1 < s[1] and x2 > s[0]]
        merged = (
            min([x1] + [s[0] for s in overlapping]),
            max([x2] + [s[1] for s in overlapping]),
        )
        spans = [s for s in spans if s not in overlapping] + [merged]
    spans.sort()
    if len(spans) < cluster_count:
        return None
    while len(spans) > cluster_count:
        gaps = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
        i = gaps.index(min(gaps))
        spans[i : i + 2] = [(spans[i][0], max(spans[i][1], spans[i + 1][1]))]
    clusters: list[list[_StreamWord]] = [[] for _ in spans]
    for word in words:
        x1 = word.bbox_ppm[0]
        index = max(i for i, span in enumerate(spans) if span[0] <= x1)
        clusters[index].append(word)
    return clusters


def _assignment_key(assignment: dict[int, list[_StreamWord]]) -> tuple[Any, ...]:
    """What an assignment *is*, for telling two of them apart: which
    words each cell gets. Two derivations that cut the same table in a
    different order reach the same assignment and must not count as an
    ambiguity — only a different assignment is one."""
    return tuple(
        sorted(
            (key, tuple(word.bbox_ppm for word in cluster))
            for key, cluster in assignment.items()
        )
    )


def _characters(cells: list[_BlockCell]) -> Counter[str]:
    """Every character these cells' paragraphs claim."""
    counts: Counter[str] = Counter()
    for cell in cells:
        for paragraph in cell.paragraphs:
            for word in paragraph:
                counts.update(word[0])
    return counts


def _characters_left_of(
    words: list[_StreamWord], positions: list[int]
) -> tuple[dict[int, Counter[str]], Counter[str]]:
    """Character counts left of each candidate cut, and of the block.

    Accumulated in one pass over the words sorted by their right edge,
    so that pairing a grid boundary with a corridor costs an alphabet
    rather than a word list. Counting per candidate instead made an
    exhausted search cost 2 ms per block on a 40-column table — the
    refusal was slower than the resolution.
    """
    left_of: dict[int, Counter[str]] = {}
    running: Counter[str] = Counter()
    by_right_edge = sorted(words, key=lambda word: word.bbox_ppm[2])
    index = 0
    for position in positions:
        while index < len(by_right_edge) and by_right_edge[index].bbox_ppm[2] <= position:
            running.update(by_right_edge[index].key)
            index += 1
        left_of[position] = running.copy()
    while index < len(by_right_edge):
        running.update(by_right_edge[index].key)
        index += 1
    return left_of, running


def _peel_rows(
    words: list[_StreamWord],
    cells: list[_BlockCell],
    columns: tuple[int, int],
    search: _Search,
) -> dict[int, list[_StreamWord]] | None:
    """Peel leading runs of rows the AST closes, one after another, and
    hand what is left back to :func:`_resolve_block`; None if the first
    cut does not resolve.

    A prefix of rows is *closed* when no cell of it is merged into a
    later row. Then the block's cells fall in two without cutting one in
    half, and the prefix's glyphs are a contiguous run at the head of
    the block's stream — a row band is emitted before the next one. The
    run is bounded by character multiset, not by sequence, because
    inside a band the stream interleaves the wrapped lines of the
    band's cells.

    A ``row_span`` cell is exactly what leaves no closed prefix: its
    text belongs to two bands at once, which is why "one row is one
    contiguous segment" lost both rows of every span table. Then the
    block has to be cut the other way, by :func:`_split_columns`.

    Iterative on purpose. Recursing once per row overflowed the
    interpreter stack on a table of 862 rows — the tallest in a
    600-document sample of the real corpus — and a `RecursionError`
    takes the whole sidecar down instead of refusing one table. Each
    cut is now committed as soon as its own head resolves, rather than
    only once the entire remainder does; that is what the row-at-a-time
    matcher this replaced did too, and no document of the 900-document
    cohort or of the fixtures changes (measured 2026-09-08). It is a
    narrowing all the same: a prefix that resolves and then leads to a
    dead end is no longer retried at a larger size.
    """
    peeled: dict[int, list[_StreamWord]] = {}
    while True:
        cut: tuple[int, list[_BlockCell], dict[int, list[_StreamWord]]] | None = None
        for last in sorted({cell.row_start for cell in cells})[:-1]:
            head = [cell for cell in cells if cell.row_start <= last]
            if any(cell.row_end > last + 1 for cell in head):
                # A merged cell reaches past the cut, so this is not a band.
                # PROVEN, having been marked unproven here for a day: the
                # 900-document cohort cannot tell this line from letting the
                # peel through, but 1 500 generated tables can — 158 of them
                # differ, and dropping it turns a refusal into a WRONG box
                # where a stray glyph repeats a merged cell's own word
                # (`test_align_document_merged_cell_is_not_peeled_across_a_
                # stray_glyph`, the smallest of 4 000 such tables). It is
                # also the only place ``row_span`` reaches the assignment.
                # Read the cost with it: against ground truth, dropping it
                # boxes 5 895 cells rather than 4 430 while wrong boxes go
                # 20 -> 29, so it is refusing about a quarter of the cells it
                # could place to avoid nine wrong ones. That trade is
                # measured on a generator, not on the real corpus.
                continue
            target = "".join(w[0] for c in head for p in c.paragraphs for w in p)
            segment = _find_row_segment(target, words, 0)
            if segment is None:
                continue
            resolved_head = _resolve_block(words[: segment[1]], head, columns, search)
            if resolved_head is None:
                continue
            cut = (
                segment[1],
                [cell for cell in cells if cell.row_start > last],
                resolved_head,
            )
            break
        if cut is None:
            if not peeled:
                return None
            # Nothing left to peel, and at least one band was: the
            # remainder is a strictly smaller block, so handing it back
            # cannot come round to this one.
            rest = _resolve_block(words, cells, columns, search)
            return None if rest is None else {**peeled, **rest}
        end, cells, resolved_head = cut
        peeled.update(resolved_head)
        words = words[end:]


def _split_columns(
    words: list[_StreamWord],
    cells: list[_BlockCell],
    columns: tuple[int, int],
    search: _Search,
) -> dict[int, list[_StreamWord]] | None:
    """Split the block at a grid boundary no cell of it crosses, cut at
    one of the corridors of *these* words, or None.

    Corridors are recomputed for every block, which is what makes the
    boundary a merged cell hides appear once that cell is peeled off:
    in a span table the `col0|col1` gutter is covered by the colspan
    cell in 50 documents of 50, and empty in all 50 once its words are
    gone. It is also why a row whose text overflows its own gutter
    closes that gutter for its band only, and not for the table.

    The cut is chosen by verifying it. Counting corridors does not
    identify it — after peeling, the remainder of a span table carries
    one corridor in 32 documents, two in 12, three in 3, four in 2 and
    five in 1, and the true boundary is only ever *among* them — while
    exactly one candidate split spells all the cells, in 50 of 50, and
    it is always the true one. Two different assignments that both
    spell them mean the cells cannot be told apart, and none means they
    are not there; both refuse, and refusing is a typed mismatch rather
    than a best guess.
    """
    positions = sorted({start for start, _end in _corridors(words)})
    left_of, whole = _characters_left_of(words, positions)
    solutions: dict[tuple[Any, ...], dict[int, list[_StreamWord]]] = {}
    for boundary in range(columns[0] + 1, columns[1]):
        if any(cell.column_start < boundary < cell.column_end for cell in cells):
            continue
        search.spend(len(positions))
        left_cells = [cell for cell in cells if cell.column_end <= boundary]
        right_cells = [cell for cell in cells if cell.column_start >= boundary]
        left_needs, right_needs = _characters(left_cells), _characters(right_cells)
        for position in positions:
            # A boundary is paired only with the corridors that could
            # serve it. Every cell is verified by exact character
            # equality inside its own cluster, so each of its characters
            # has to be on its side of the cut: a necessary condition,
            # rejecting no candidate that verification would have
            # accepted, and the difference between a 12-column table
            # resolving and exhausting the budget.
            if left_needs - left_of[position]:
                continue
            if right_needs - (whole - left_of[position]):
                continue
            # a corridor is empty, so every word lies wholly left or
            # wholly right of it: the cut needs no midpoint, no
            # tolerance and no width
            left = [w for w in words if w.bbox_ppm[2] <= position]
            right = [w for w in words if w.bbox_ppm[2] > position]
            resolved_left = _resolve_block(left, left_cells, (columns[0], boundary), search)
            if resolved_left is None:
                continue
            resolved_right = _resolve_block(
                right, right_cells, (boundary, columns[1]), search
            )
            if resolved_right is None:
                continue
            merged = {**resolved_left, **resolved_right}
            solutions[_assignment_key(merged)] = merged
            if len(solutions) > 1:
                return None  # ambiguous; counting further is not needed
    return next(iter(solutions.values())) if len(solutions) == 1 else None


def _resolve_block(
    words: list[_StreamWord],
    cells: list[_BlockCell],
    columns: tuple[int, int],
    search: _Search,
) -> dict[int, list[_StreamWord]] | None:
    """Which of ``words`` belongs to each cell of one block of a table,
    or None if the block cannot be told apart.

    A block is a contiguous range of the table's grid columns, the cells
    that live wholly inside it (AST row-major order) and the words that
    fall in it. It is resolved by peeling: cut it in two — after a
    closed run of rows (:func:`_peel_rows`) or at a grid boundary no
    cell crosses (:func:`_split_columns`) — and resolve both halves the
    same way, until a block holds a single cell and the cell's own text
    verifies it. Every cut is verified by exact character equality.

    That is NOT the same as "a wrong assignment can only surface as a
    typed mismatch", which the plan, this docstring and several commit
    messages all claimed. Exact character equality cannot tell two cells
    of identical text apart, so where the stream lists them in the other
    order the boxes come out exchanged and verify perfectly. On the
    column axis :func:`_split_columns` catches it by counting verifying
    assignments; on the row axis counting is not available, and
    :func:`_rows_in_document_order` catches it afterwards by checking
    the assignment against the grid instead of against the text. What
    holds is narrower and worth stating exactly: a box is always some
    word that spells the cell, and an assignment that contradicts the
    AST's row order is refused.

    Each cut hands both halves strictly fewer cells than the block had
    (a closed row prefix is non-empty and leaves a non-empty tail; a
    boundary strictly inside the block has a cell touching each of its
    ends, and no cell crosses it), so the recursion is on a strictly
    decreasing positive integer and terminates in at most one step per
    cell. ``search`` carries the memo that keeps the *derivations* from
    growing Catalan in the column count, and the budget that bounds the
    work; spending the budget raises :class:`_SearchExhausted`, which
    refuses the whole table rather than letting a half-run search look
    like a verdict.
    """
    # ``columns`` is part of the key because a block is a column RANGE,
    # not just its cells: the same cells and words can sit in a wider
    # range and offer a boundary this one does not. UNPROVEN — dropping
    # it differs on 0 of 4 000 generated tables (mutated 2026-09-08) and
    # no test sees it, so read it as a statement of what a block is, not
    # as a caught error.
    key = (columns, tuple(cell.key for cell in cells), tuple(words))
    if key in search.memo:
        return search.memo[key]
    search.spend(len(words) + 1)
    if len(cells) == 1:
        resolved = (
            None if _cell_spans(words, cells[0].paragraphs) is None
            else {cells[0].key: words}
        )
    else:
        # Uniqueness is enforced on the column axis only. `_peel_rows`
        # takes the first row prefix that resolves and a peel is
        # preferred to any split, neither compared against the other,
        # where `_split_columns` refuses two assignments that both
        # verify.
        #
        # PROVEN to matter, contrary to what this comment said before:
        # two cells of one column with identical text and a stream that
        # lists them in the other order come back with their boxes
        # exchanged and no issue raised. What rules the exchange out is
        # the grid, in :func:`_rows_in_document_order`.
        #
        # Counting row peels was tried and REJECTED ON ITS COST, which
        # is not the reason recorded here earlier: coverage collapses
        # from 4 430 boxed cells to 2 412 (-45 %) against ground truth.
        # The earlier reason — that counting would break the repainted
        # header, which needs the FIRST of two identical bands — is NOT
        # what the evidence shows: with counting in place the repaint
        # fixture still puts `h0p` on page 0, because counting the
        # peel's candidate row prefixes never sees the duplication at
        # all. Counting whole assignments, the way the column axis does,
        # would see it; that variant is unmeasured.
        #
        # The ORDER of the two cuts is still unproven — swapping it
        # leaves the suite green and the cohort byte-identical (mutated
        # 2026-09-08), and 1 500 generated documents agree.
        resolved = _peel_rows(words, cells, columns, search)
        if resolved is None:
            resolved = _split_columns(words, cells, columns, search)
    search.memo[key] = resolved
    return resolved


def _match_sequential(
    group: list[_Paragraph],
    words_per_paragraph: list[list[tuple[str, int, int]]],
    pdf_words: list[_StreamWord],
    pos: int,
    issues: list[Issue],
    consumed: bytearray,
    index: _StreamIndex | None = None,
    next_texts: list[str] | None = None,
    global_owned_ids: set[int] | None = None,
    page_break_split: bool = False,
    tab_segments: bool = False,
) -> int:
    """Single-paragraph group: exact sequential word match, resyncing
    forward over orphan PDF words (e.g. list marker glyphs).

    ``page_break_split`` (policy `v16-align.v1`, A9): a paragraph no
    contiguous run spells may still be spelled by one run per page, each
    continuing on the next page after that page's furniture
    (:func:`_spans_across_page_breaks`).

    ``tab_segments`` (policy `v16-align.v1`, A16): a paragraph with tabs
    that neither rule spells may be spelled segment by segment
    (:func:`_span_tab_segments`)."""
    paragraph, ast_words = group[0], words_per_paragraph[0]
    texts = [w[0] for w in ast_words]
    span = _find_char_span(texts, pdf_words, pos, index, next_texts)
    if span is None and next_texts:
        # Bounded last resort. The strict resync gave up — the paragraph
        # occurs more than once in the window, or its corroboration
        # failed because the following group's opening is non-contiguous
        # (list markers, page furniture). The next group's opening words
        # still delimit the region where this paragraph can be: if its
        # text is spelled exactly once inside [pos, next_start) and that
        # span is unclaimed, it is the one. Two candidates inside the
        # region mean no box, exactly as in the strict path — the region
        # narrows the search, it does not pick a side.
        next_start = _locate_next_group_opening(next_texts, pdf_words, pos)
        if next_start is not None:
            target = "".join(texts)
            candidates: list[tuple[int, int]] = []
            starts = (
                [s for s in index.candidate_starts(target, pos) if s < next_start]
                if index is not None
                else range(pos, next_start)
            )
            for start in starts:
                found = _span_at(target, pdf_words, start)
                if found is None or found[1] > next_start:
                    continue
                if any(consumed[found[0] : found[1]]):
                    continue
                candidates.append(found)
            if len(candidates) == 1:
                span = candidates[0]
    if span is None and page_break_split:
        pieces = _spans_across_page_breaks(texts, pdf_words, pos, consumed)
        if pieces is not None:
            first_start, first_end = pieces[0]
            skipped = [w for i, w in enumerate(pdf_words[pos:first_start], pos) if not consumed[i]]
            if skipped:
                issues.append(_orphan_issue(skipped, paragraph.path))
            words = [w for start, end in pieces for w in pdf_words[start:end]]
            _consume(paragraph, ast_words, words, global_owned_ids=global_owned_ids)
            # the furniture between the pieces stays unconsumed: it is
            # reported as orphan text at the end unless a story claims it
            consumed[pos:first_end] = b"\1" * (first_end - pos)
            for start, end in pieces[1:]:
                consumed[start:end] = b"\1" * (end - start)
            return pieces[-1][1]
    if span is None and tab_segments and "\t" in paragraph.text and index is not None:
        located = _span_tab_segments(paragraph, ast_words, pdf_words, pos, index, global_owned_ids)
        if located is not None:
            main_start, main_end = located[0]
            skipped = [w for i, w in enumerate(pdf_words[pos:main_start], pos) if not consumed[i]]
            if skipped:
                issues.append(_orphan_issue(skipped, paragraph.path))
            for segment, (start, end) in located[1]:
                _consume(
                    paragraph, segment, pdf_words[start:end], global_owned_ids=global_owned_ids
                )
                consumed[start:end] = b"\1" * (end - start)
            consumed[pos:main_end] = b"\1" * max(main_end - pos, 0)
            return max([main_end] + [end for _seg, (start, end) in located[1] if start >= pos])
    if span is None:
        stream = [w.text for w in pdf_words[pos : pos + len(texts)]]
        raw_texts = [
            paragraph.text[char_start:char_end] for _key, char_start, char_end in ast_words
        ]
        issues.append(_mismatch_issue(paragraph, raw_texts, stream))
        return pos
    found, end = span
    skipped = [w for i, w in enumerate(pdf_words[pos:found], pos) if not consumed[i]]
    if skipped:
        # only the words nothing has accounted for: a table located
        # between its anchors takes words from anywhere in the stream, so
        # some of what lies between ``pos`` and ``found`` already carries
        # a box. Reporting those as orphan text would fail a gate on a
        # document that is fully aligned.
        issues.append(_orphan_issue(skipped, paragraph.path))
    _consume(paragraph, ast_words, pdf_words[found:end], global_owned_ids=global_owned_ids)
    consumed[pos:end] = b"\1" * (end - pos)
    return end


#: work bound of one `_spans_across_page_breaks` search (words compared)
#: and its page bound (recursion depth); an exhausted search matches
#: nothing, like a failed one
_PAGE_BREAK_SPLIT_BUDGET = 2_000_000
_PAGE_BREAK_SPLIT_MAX_PAGES = 400


def _spans_across_page_breaks(
    texts: list[str],
    pdf_words: list[_StreamWord],
    pos: int,
    consumed: bytearray,
) -> list[tuple[int, int]] | None:
    """One ``(start, end)`` run per page when the paragraph keys ``texts``
    are spelled by a run that starts within `_RESYNC_WINDOW` of ``pos``
    and stops matching after a word on page p, continued -- within
    another `_RESYNC_WINDOW` words, all on pages p and p+1 -- by a run on
    page p+1, and so on until a run on the last page spells the rest
    exactly (A9; a stenogram paragraph of 254 396 characters spans ~70
    pages, iteration 2).

    Every run is greedy (it takes every word the paragraph continues
    with, across a page end that has no furniture too) and every run after
    the first starts on the page after the previous run's last word; a
    continuation two pages later is refused. A run that leads nowhere is
    backtracked to the next start on the same page. The break is not
    required to sit at a word boundary of the AST text, but the pieces are
    exact characters, so a word
    hyphenated across the break does not match (as it does not within a
    page)."""
    target = "".join(texts)
    if not target:
        return None
    n = len(pdf_words)
    budget = [_PAGE_BREAK_SPLIT_BUDGET]

    def greedy(start: int, offset: int) -> tuple[int, int]:
        i = start
        while i < n and offset < len(target):
            key = pdf_words[i].key
            if not target.startswith(key, offset):
                break
            offset += len(key)
            i += 1
        budget[0] -= i - start + 1
        return i, offset

    def rest_from(i: int, offset: int, depth: int) -> list[tuple[int, int]] | None:
        # the run ending at i-1 stopped short of the paragraph's end
        if depth >= _PAGE_BREAK_SPLIT_MAX_PAGES:
            return None
        page = pdf_words[i - 1].page_index
        for j in range(i, min(n, i + _RESYNC_WINDOW + 1)):
            if budget[0] <= 0 or pdf_words[j].page_index > page + 1:
                return None
            if pdf_words[j].page_index != page + 1 or consumed[j]:
                continue
            end, reached = greedy(j, offset)
            if reached == offset:
                continue
            if reached == len(target):
                return [(j, end)]
            if end >= n:
                continue
            later = rest_from(end, reached, depth + 1)
            if later is not None:
                return [(j, end), *later]
        return None

    for start in range(pos, min(n, pos + _RESYNC_WINDOW + 1)):
        if consumed[start]:
            continue
        i, offset = greedy(start, 0)
        if offset == 0 or offset == len(target) or i >= n:
            continue
        rest = rest_from(i, offset, 1)
        if rest is not None:
            return [(start, i), *rest]
        if budget[0] <= 0:
            return None
    return None


#: A16: the longest tab segment must carry at least this many keyed
#: characters to be located on its own (a short one is no evidence)
_TAB_MAIN_MIN_CHARS = 12
#: A16: a neighbouring segment may overlap the one beside it horizontally
#: by this much (page ppm) and still count as beside it
_TAB_BESIDE_SLACK_PPM = 2_000


def _tab_split(
    paragraph: _Paragraph, ast_words: list[tuple[str, int, int]]
) -> list[list[tuple[str, int, int]]]:
    segments: list[list[tuple[str, int, int]]] = [[]]
    previous_end = 0
    for word in ast_words:
        if segments[-1] and "\t" in paragraph.text[previous_end : word[1]]:
            segments.append([])
        segments[-1].append(word)
        previous_end = word[2]
    return [segment for segment in segments if segment]


def _segment_beside(
    segment: list[tuple[str, int, int]],
    pdf_words: list[_StreamWord],
    index: _StreamIndex,
    neighbour: tuple[int, int],
    left: bool,
    unavailable: set[int],
) -> tuple[int, int] | None:
    """The run of words spelling ``segment`` that sits on the same visual
    line as the ``neighbour`` run, immediately left (``left``) or right of
    it, nearest first; words in ``unavailable`` (owned elsewhere or taken
    by another segment) never count."""
    ref = pdf_words[neighbour[0]] if left else pdf_words[neighbour[1] - 1]
    y0, y1 = ref.bbox_ppm[1], ref.bbox_ppm[3]
    target = "".join(key for key, _start, _end in segment)
    best: tuple[int, tuple[int, int]] | None = None
    for start in index.candidate_starts(target, 0):
        span = _span_at(target, pdf_words, start)
        if span is None:
            continue
        words = pdf_words[span[0] : span[1]]
        if any(id(w) in unavailable or w.page_index != ref.page_index for w in words):
            continue
        edge = words[-1] if left else words[0]
        if not (edge.bbox_ppm[1] < y1 and edge.bbox_ppm[3] > y0):
            continue
        gap = (
            ref.bbox_ppm[0] - edge.bbox_ppm[2] if left else edge.bbox_ppm[0] - ref.bbox_ppm[2]
        )
        if gap < -_TAB_BESIDE_SLACK_PPM:
            continue
        if best is None or gap < best[0]:
            best = (gap, span)
    return None if best is None else best[1]


def _span_tab_segments(
    paragraph: _Paragraph,
    ast_words: list[tuple[str, int, int]],
    pdf_words: list[_StreamWord],
    pos: int,
    index: _StreamIndex,
    owned: set[int] | None,
) -> tuple[tuple[int, int], list[tuple[list[tuple[str, int, int]], tuple[int, int]]]] | None:
    """A16: ``(main span, [(segment, span)] in paragraph order)`` when the
    paragraph's tab segments are each spelled: the longest one contiguously
    (`_find_char_span` from ``pos``), then outward from it every other
    segment on the same visual line beside its neighbour
    (`_segment_beside`). pdftotext -layout reads a hanging label column
    ahead of the text ("(d) (2) any building ... For the purposes ...") and
    lets footer words cut a label off its line ("(1) page 2 In these
    regulations"); 331 of the 377 unmatched body paragraphs of the
    iteration-2 sample have a tab."""
    segments = _tab_split(paragraph, ast_words)
    if len(segments) < 2:
        return None
    main = max(range(len(segments)), key=lambda i: sum(len(k) for k, _s, _e in segments[i]))
    keys = [key for key, _start, _end in segments[main]]
    if sum(len(key) for key in keys) < _TAB_MAIN_MIN_CHARS:
        return None
    span = _find_char_span(keys, pdf_words, pos, index)
    unavailable = set(owned or ())
    if span is None or span[0] == span[1] or any(
        id(w) in unavailable for w in pdf_words[span[0] : span[1]]
    ):
        return None
    located = {main: span}
    unavailable |= {id(w) for w in pdf_words[span[0] : span[1]]}
    order = [*range(main - 1, -1, -1), *range(main + 1, len(segments))]
    for i in order:
        left = i < main
        found = _segment_beside(
            segments[i], pdf_words, index, located[i + 1 if left else i - 1], left, unavailable
        )
        if found is None:
            return None
        located[i] = found
        unavailable |= {id(w) for w in pdf_words[found[0] : found[1]]}
    return span, [(segments[i], located[i]) for i in range(len(segments))]


def _anchor_after(
    pdf_words: list[_StreamWord],
    pos: int,
    index: _StreamIndex | None,
    next_texts: list[str] | None,
) -> int | None:
    """Stream index of the first word of the group that follows, or None
    if there is no following group or it cannot be located.

    None widens the region to the end of the document rather than
    narrowing it: a bracket that absorbs too much yields orphans and a
    typed mismatch, while one that cuts a block short would drop its
    words with nothing to say so.
    """
    if not next_texts:
        return None
    span = _find_char_span(next_texts, pdf_words, pos, index)
    return span[0] if span is not None else None


def _row_texts(cells: list[_BlockCell]) -> list[str]:
    """The text of each row band of a table, in AST order."""
    bands: dict[int, list[str]] = {}
    for cell in cells:
        for paragraph in cell.paragraphs:
            bands.setdefault(cell.row_start, []).extend(word[0] for word in paragraph)
    return ["".join(bands[row]) for row in sorted(bands)]


def _skips_a_repainted_band(
    pdf_words: list[_StreamWord], pos: int, found: int, row_texts: list[str]
) -> bool:
    """Is the table at ``found`` a REPAINT of the rows just skipped?

    `repeat_header` repaints the leading rows at the top of the
    continuation page, so the region that begins at the table's first
    glyph carries those characters twice and fails the exact-multiset
    test, and the search slides to the second occurrence. The table then
    takes its box on the continuation page and the original becomes
    orphan text — the reverse of what the plan recorded as the cost of
    deferring repeated headers, and a wrong page for `page_evidence`,
    the mask and the target of every node in those rows.

    Three things have to hold, and each of them rules out a shape that
    was measured going wrong:

    * the skipped run contains whole leading bands, by character
      multiset, in any order — a band's cells interleave, and a
      repainted word may be split by the renderer differently than the
      original (`head` -> `he` `ad`), so neither sequence nor tokens are
      shared, only characters;
    * what is left of the skipped run after those bands is no bigger
      than the bands themselves. Containment alone is arithmetic on a
      multiset: over 150 table-bearing real documents this fired three
      times, on skips of 3 253 to 4 230 characters against bands of 38
      to 54 — 77 to 105 times the band — and not one was a repaint. Any
      long run contains a short band. The page furniture of a break, a
      footer or a running head, is the size of a band, not a hundred
      times it;
    * the repaint is on a later page than the copy it repeats, because
      that is when Word paints one: at a page break. Without this a
      stray glyph that equals a short first row band — a caption
      reading `Total` above a table whose first row is `Total` — made
      the veto fire and the cell take the stray's box.

    The long skips this no longer fires on are real and want refusing;
    :func:`_located_past_what_follows` is what refuses them,
    under its own name.
    """
    if found >= len(pdf_words) or found <= pos:
        return False
    skipped: Counter[str] = Counter()
    for word in pdf_words[pos:found]:
        skipped.update(word.key)
    if not skipped:
        return False
    if pdf_words[found].page_index <= min(w.page_index for w in pdf_words[pos:found]):
        return False
    band: Counter[str] = Counter()
    for row_text in row_texts:
        band.update(row_text)
        leftover = skipped - band
        if not (band - skipped) and sum(leftover.values()) <= sum(band.values()):
            return True
    return False


def _located_past_what_follows(found: int, anchor: int | None) -> bool:
    """Does locating the table here put it AFTER the group that comes
    next in the document?

    Localisation slides forward over words no AST node accounts for — a
    list marker, or a block the aligner could not match, which is why
    the slide is not bounded by distance: a table's characters are
    contiguous wherever the table is, and refusing a far one loses the
    whole document suffix (`test_aligner_resync`).

    What a far location may not do is land past the content that comes
    after it. Measured on a worksheet whose every question is a table of
    the same shape and vocabulary: the search slid over 3 253 to 4 230
    characters — 77 to 105 times the table's own first band — to land on
    ANOTHER question's table, several groups further down. Refusing
    those wins 385 nodes of 864, because the table's own words are then
    taken between its anchors, where they are.

    Ordering, not distance: the same rule `_find_char_span` already
    applies to a paragraph when it corroborates a far resync with the
    next group's words. There is no constant in it, and a table with
    nothing after it (``anchor`` None) may still resync as far as it
    likes.
    """
    # strictly past: where the two coincide, the group that follows was
    # located ON this table's own words — a continuation table repeating
    # this one's header row does exactly that — and the anchor says
    # nothing about where this table is. Measured: `>=` cost a real
    # document its whole coverage, 108/108 down to 48/108.
    return anchor is not None and found > anchor


def _reanchor_past_duplicate(
    pdf_words: list[_StreamWord],
    anchor: int | None,
    segment: tuple[int, int],
    next_texts: list[str] | None,
    index: _StreamIndex | None,
) -> int | None:
    """Re-point a stale anchor at the occurrence that follows the table.

    ``_anchor_after`` takes the FIRST occurrence of the next group's
    text after ``pos``. That is the group only when nothing earlier in
    the stream repeats it. A cover line or running head equal to a
    later heading lands the anchor before a table the search located
    correctly, and the location is then refused against a duplicate
    that anchors nothing — the table's words are taken between the
    duplicate and ``pos`` and the document loses its whole table
    (``bundle_90502e54``: the cover title equals the heading after a
    4x2 table; the stale anchor cost all eight cells, 46/62 instead of
    62/62). If the text also occurs past the located segment, that
    occurrence is the group that follows: the location lands before
    it, exactly where a table that precedes its group belongs. If it
    does not, the location slid past the following group and the
    original anchor stands (:func:`_located_past_what_follows` keeps
    its refusal, measured on the worksheet at 385 of 864 nodes).
    """
    if anchor is None or not next_texts:
        return anchor
    later = _find_char_span(next_texts, pdf_words, segment[1], index)
    if later is None:
        return anchor
    return later[0]


def _reanchor_past_own_echo(
    pdf_words: list[_StreamWord],
    pos: int,
    anchor: int,
    next_texts: list[str] | None,
    index: _StreamIndex | None,
    own_chars: int,
) -> int:
    """Re-point an anchor that sits inside the table's own span.

    ``_anchor_after`` takes the FIRST occurrence of the next group's
    text after ``pos``. That is the group only when the stream has
    already carried the table's own characters by the time it reaches
    it. A table whose words are not a run of the stream (a footnote
    body interleaves its rows) has no located segment, and its cells
    may repeat the next group's text: the anchor then lands inside the
    table, the bracket that takes the table between its anchors ends
    where the table's own rows are still going, and every row below it
    is lost (``bundle_59a26b49``: form cells ending "...Part I of
    Annex" and the group after the table opening on "Annex"; the stale
    anchor cost 17 rows, 34 nodes, 72/106 instead of 106/106). The
    first later occurrence at which the stream has carried the table's
    own characters is the group that follows. Until one is found the
    original anchor stands: this narrows a region, and a wrong
    re-anchor would lose more, not less.
    """
    if not next_texts:
        return anchor
    carried = sum(len(word.key) for word in pdf_words[pos:anchor])
    if carried >= own_chars:
        return anchor
    target = "".join(next_texts)
    if index is not None:
        candidates = index.candidate_starts(target, anchor)
    else:
        candidates = list(range(anchor, len(pdf_words)))
    cursor = anchor
    for cand in candidates:
        span = _span_at(target, pdf_words, cand)
        if span is None:
            continue
        carried += sum(len(w.key) for w in pdf_words[cursor:cand])
        cursor = cand
        if carried >= own_chars:
            return span[0]
    return anchor


def _located_segment_uses_repainted_head(
    pdf_words: list[_StreamWord],
    pos: int,
    found: int,
    resume: int,
    row_texts: list[str],
) -> bool:
    """Whether a located table segment skipped a leading band and then
    used a later copy of that band instead.

    A ``repeat_header`` band can make the table's exact character
    multiset fit in a window that starts AFTER the table's own first
    band: the original band plus the continuation plus the repaint has
    one band too many, while the continuation plus the repaint has
    exactly the table's characters. The segment is then anchored on the
    repaint, the original band becomes orphan text, and the rows between
    them sit above the band the matcher finally places.

    This is the same-page twin of :func:`_skips_a_repainted_band`: the
    skipped band and the segment's start may sit on one page, but the
    copy the segment actually uses must be a later repaint. A stray
    caption equal to the first band does not fire, because the located
    segment starts with the table's own first band.
    """
    if found <= pos or found >= len(pdf_words) or resume <= found:
        return False
    skipped = pdf_words[pos:found]
    segment = pdf_words[found:resume]
    if not skipped or not segment:
        return False
    skipped_counts: Counter[str] = Counter()
    segment_counts: Counter[str] = Counter()
    for word in skipped:
        skipped_counts.update(word.key)
    for word in segment:
        segment_counts.update(word.key)
    skipped_min_page = min(word.page_index for word in skipped)
    band: Counter[str] = Counter()
    leading: list[str] = []
    for row_text in row_texts:
        leading.append(row_text)
        band.update(row_text)
        if band - skipped_counts:
            continue
        leftover = skipped_counts - band
        if sum(leftover.values()) > sum(band.values()):
            continue
        for candidate_row in leading:
            row = Counter(candidate_row)
            if not row or row - skipped_counts or row - segment_counts:
                continue
            occurrence = _find_row_segment(candidate_row, segment, 0, window=None)
            if occurrence is None or occurrence[0] == 0:
                continue
            repaint_words = segment[occurrence[0] : occurrence[1]]
            if repaint_words and min(
                word.page_index for word in repaint_words
            ) > skipped_min_page:
                return True
    return False


def _row_top(
    cells: list[_BlockCell], assignment: dict[int, list[_StreamWord]], row: int
) -> tuple[int, int] | None:
    """The top of the words ``assignment`` already gave ``row``, if any."""
    top: tuple[int, int] | None = None
    for cell in cells:
        if cell.row_start != row:
            continue
        cluster = assignment.get(cell.key)
        if not cluster:
            continue
        spans = _cell_spans(cluster, cell.paragraphs)
        if spans is None:
            continue
        for start, end in spans:
            for word in cluster[start:end]:
                position = (word.page_index, word.bbox_ppm[1])
                if top is None or position < top:
                    top = position
    return top


def _row_segment_spells_cells(
    segment_words: list[_StreamWord], row_cells: list[_BlockCell]
) -> bool:
    """Whether one contiguous segment can be the row's own band."""
    if not row_cells:
        return False
    clusters = _x_clusters_stream(segment_words, len(row_cells))
    if clusters is None:
        return False
    return all(
        _cell_spans(cluster, cell.paragraphs) is not None
        for cell, cluster in zip(row_cells, clusters)
    )


def _row_segment_has_rival(
    target: str,
    pdf_words: list[_StreamWord],
    start: int,
    end_limit: int | None,
    row_cells: list[_BlockCell],
) -> bool:
    """Another segment, after ``start``, that could be the same row."""
    if end_limit is not None and start >= end_limit:
        return False
    search = _Search(remaining=_BAND_SEARCH_BUDGET, budget=_BAND_SEARCH_BUDGET)
    probe = start
    try:
        while True:
            window = (end_limit - probe) if end_limit is not None else None
            segment = _find_row_segment(
                target, pdf_words, probe, window=window, search=search
            )
            if segment is None:
                return False
            if _row_segment_spells_cells(
                pdf_words[segment[0] : segment[1]], row_cells
            ):
                return True
            probe = segment[0] + 1
            if end_limit is not None and probe >= end_limit:
                return False
    except _SearchExhausted:
        # the recheck cannot finish: refusing is the fail-closed reading
        return True


def _match_table_row_fallback(
    cells: list[_BlockCell],
    group: list[_Paragraph],
    words_per_paragraph: list[list[tuple[str, int, int]]],
    pdf_words: list[_StreamWord],
    search_start: int,
    search_end: int | None,
    issues: list[Issue],
    consumed: bytearray,
    assignment: dict[int, list[_StreamWord]],
    index: _StreamIndex | None = None,
    next_texts: list[str] | None = None,
) -> int:
    """Row-by-row fallback for tables the whole-table matcher and band
    fallback cannot resolve.

    Each row's words are contiguous in the PDF stream (LibreOffice renders
    a row's cells as a horizontal band), so searching for one row at a time
    succeeds where the whole-table search cannot. The search is bounded to
    the table's own word range ``[search_start, search_end)``: the rows are
    PART of the table, so they sit inside that range — searching from the
    range's end (the old behaviour) walked straight past the whole table.

    A row is placed only as a closed band of its own: a row that still
    holds a merged cell is skipped, a row whose candidate top lies above
    the row before it is skipped, and where a row above it failed the
    candidate must be the only segment that spells it. Every placement is
    re-checked against :func:`_rows_in_document_order` before it sticks.
    """
    if not cells:
        return search_start
    if search_end is not None and search_start >= search_end:
        return search_start

    rows_by_row: dict[int, list[_BlockCell]] = {}
    for cell in cells:
        rows_by_row.setdefault(cell.row_start, []).append(cell)

    new_pos = search_start
    resolved = False
    previous_top: tuple[int, int] | None = None
    missing = 0
    for row, row_cells in sorted(rows_by_row.items()):
        if search_end is not None and new_pos >= search_end:
            break
        unresolved = [cell for cell in row_cells if cell.key not in assignment]
        if not unresolved:
            top = _row_top(cells, assignment, row)
            if top is not None:
                if previous_top is None or top < previous_top:
                    previous_top = top
                missing = 0
            continue

        placeable = [
            cell
            for cell in unresolved
            if cell.row_end <= cell.row_start + 1 and any(cell.paragraphs)
        ]
        if not placeable or any(
            cell.row_end > cell.row_start + 1 for cell in unresolved
        ):
            missing += 1
            continue

        placeable.sort(key=lambda cell: cell.column_start)
        row_words: list[str] = []
        for cell in placeable:
            for paragraph in cell.paragraphs:
                for word in paragraph:
                    row_words.append(word[0])
        target = "".join(row_words)
        if not target:
            missing += 1
            continue

        window = (search_end - new_pos) if search_end is not None else None
        segment = _find_row_segment(target, pdf_words, new_pos, window=window)
        if segment is None:
            segment = _find_char_span(row_words, pdf_words, new_pos, index, next_texts)
        if segment is None:
            missing += 1
            continue

        found, end = segment
        if found < new_pos or (search_end is not None and end > search_end):
            missing += 1
            continue

        clusters = _x_clusters_stream(pdf_words[found:end], len(placeable))
        if clusters is None:
            missing += 1
            continue

        provisional: dict[int, list[_StreamWord]] = {}
        tops: list[tuple[int, int]] = []
        ok = True
        for cell, cluster in zip(placeable, clusters):
            spans = _cell_spans(cluster, cell.paragraphs)
            if spans is None:
                ok = False
                break
            provisional[cell.key] = cluster
            for start, stop in spans:
                for word in cluster[start:stop]:
                    tops.append((word.page_index, word.bbox_ppm[1]))
        if not ok or not tops:
            missing += 1
            continue

        candidate_top = min(tops)
        if previous_top is not None and candidate_top < previous_top:
            missing += 1
            continue

        if missing and _row_segment_has_rival(
            target, pdf_words, found + 1, search_end, placeable
        ):
            missing += 1
            continue

        tentative = dict(assignment)
        tentative.update(provisional)
        if not _rows_in_document_order(cells, tentative):
            missing += 1
            continue

        if found > new_pos:
            issues.append(_orphan_issue(pdf_words[new_pos:found], group[0].path))
        assignment.update(provisional)
        for i in range(found, end):
            consumed[i] = 1
        new_pos = end
        if previous_top is None or candidate_top < previous_top:
            previous_top = candidate_top
        missing = 0
        resolved = True

    if not resolved:
        return search_start
    return new_pos


def _match_table(
    group: list[_Paragraph],
    words_per_paragraph: list[list[tuple[str, int, int]]],
    pdf_words: list[_StreamWord],
    pos: int,
    issues: list[Issue],
    consumed: bytearray,
    index: _StreamIndex | None = None,
    next_texts: list[str] | None = None,
    global_owned_ids: set[int] | None = None,
) -> int:
    """Table group: locate the table's words, then assign them to its
    cells by peeling (:func:`_resolve_block`).

    Neither a row nor a cell is a contiguous run of the stream. Within a
    row the renderer interleaves the wrapped lines of its cells; a cell
    merged across rows interleaves the rows themselves, which is why
    matching one row at a time lost both rows of every span table. The
    table is what the assignment is scoped to, and the cuts inside it
    come from AST topology and from the corridors of the words still
    unassigned — never from the reading order, which a merged cell does
    not respect.

    The segment is bounded by character multiset rather than by
    sequence, for the same reason. That search is window-bounded
    (``pos + _RESYNC_WINDOW``): unlike :func:`_find_char_span`, which
    indexes the stream by word text and can resync anywhere in the
    remainder, a multiset segment has no such index and is only looked
    for near ``pos``. Past the window the sequential fallback and then
    the anchors take over.

    A table need not be a run of the stream at all. A `col_span` cell
    that erases the middle gutter makes poppler read the whole page as
    two column flows, and the two tables of document 17 then interleave
    with each other ([5, 96] and [54, 100]), so no contiguous location
    can succeed. The table's words are taken between its anchors
    instead — everything below the last word already accounted for and
    above the first word of the next group — which is order-free, the
    way the assignment inside the table already is.
    """
    flat_texts = [w[0] for words in words_per_paragraph for w in words]
    flat_target = "".join(flat_texts)
    cells = _table_cells(group, words_per_paragraph)
    own_words = len(flat_texts)
    own_chars = sum(len(text) for text in flat_texts)
    columns = (
        min(cell.column_start for cell in cells),
        max(cell.column_end for cell in cells),
    )
    row_texts = _row_texts(cells)
    segment = _find_row_segment(flat_target, pdf_words, pos)
    if segment is None:
        segment = _find_char_span(flat_texts, pdf_words, pos, index, next_texts)
    anchor = _anchor_after(pdf_words, pos, index, next_texts)
    if segment is not None and _located_past_what_follows(segment[0], anchor):
        # The anchor may be an earlier duplicate of the next group's
        # text (a cover line repeating a later heading): if the text
        # also occurs past the located segment, that occurrence is the
        # group that actually follows and the anchor is stale. If it
        # does not, the location slid past the following group and is
        # refused as before.
        anchor = _reanchor_past_duplicate(
            pdf_words, anchor, segment, next_texts, index
        )
        if _located_past_what_follows(segment[0], anchor):
            segment = None
    if segment is not None and _skips_a_repainted_band(
        pdf_words, pos, segment[0], row_texts
    ):
        segment = None
    located_search = _Search(
        remaining=_TABLE_SEARCH_BUDGET, budget=_TABLE_SEARCH_BUDGET
    )
    located_assignment: dict[int, list[_StreamWord]] | None = None
    grid_rejected = False
    located_segment: tuple[int, int] | None = None
    if segment is not None:
        found, resume = segment
        first_found, first_resume = found, resume
        while True:
            candidate = [pdf_words[i] for i in range(found, resume)]
            try:
                located_assignment = _resolve_block(
                    candidate, cells, columns, located_search
                )
            except _SearchExhausted:
                if located_assignment is not None:
                    grid_rejected = True
                    located_assignment = {}
                else:
                    located_assignment = None
                found, resume = first_found, first_resume
                break
            if located_assignment is not None and _rows_in_document_order(
                cells, located_assignment
            ):
                break
            # A located segment that does not yield a grid-valid
            # resolution may be a shifted occurrence of the same table:
            # try the next exact segment before consuming it. If no later
            # segment resolves, the FIRST located segment is what the band
            # and row fallbacks below still receive.
            next_segment = _find_row_segment(flat_target, pdf_words, found + 1)
            if next_segment is None or (
                _skips_a_repainted_band(
                    pdf_words, pos, next_segment[0], row_texts
                )
                or _located_past_what_follows(next_segment[0], anchor)
            ):
                if located_assignment is not None:
                    grid_rejected = True
                    located_assignment = {}
                found, resume = first_found, first_resume
                break
            found, resume = next_segment
        # A located segment that skipped the table's own leading band and
        # anchored on its later repaint is not a location: consuming it
        # would orphan the original band and place the rows between it and
        # the repaint out of order. Take the table between its anchors.
        if located_assignment is None and _located_segment_uses_repainted_head(
            pdf_words, pos, found, resume, row_texts
        ):
            located_segment = None
        else:
            located_segment = (found, resume)
    if located_segment is not None:
        found, resume = located_segment
        picked = list(range(found, resume))
        # located and therefore spent, whether or not the cells resolve
        if found > pos:
            issues.append(_orphan_issue(pdf_words[pos:found], group[0].path))
        consumed[pos:resume] = b"\1" * (resume - pos)
        bracketed = False
        search_start, search_end = found, resume
    else:
        if anchor is not None:
            # The unlocated table is bracketed by this anchor, and the
            # anchor may be the table's own cell echoing the next
            # group's text: the stream has not carried the table's
            # characters by it, so it sits inside the table and would
            # end the bracket at a mid-table row.
            anchor = _reanchor_past_own_echo(
                pdf_words, pos, anchor, next_texts, index, own_chars
            )
        picked = _bracketed_words(
            pdf_words,
            consumed,
            pdf_words[pos - 1] if pos > 0 else None,
            pdf_words[anchor] if anchor is not None else None,
        )
        resume = pos
        bracketed = True
        # the rows sit inside the bracketed region, not past it; never
        # search backwards of the current position
        search_start = max(picked[0], pos) if picked else pos
        search_end = picked[-1] + 1 if picked else None

    table_words = [pdf_words[i] for i in picked]
    # UNANCHORED only: with no anchor `_bracketed_words` takes every
    # unconsumed word to the end of the document, and a single cell (or
    # a column split's terminal cell) that resolves via `_cell_spans`
    # accepts the WHOLE candidate list, not just the span it verified —
    # so a dot-leader paragraph the aligner cannot locate lets one table
    # eat the rest of the stream (class A). The whole-table attempt is
    # therefore bounded to the first `own_words * _BRACKET_SLACK` words
    # here; the ANCHORED case is already bounded by the located anchor,
    # and the per-row band fallback below stays on the FULL bracket
    # regardless — its own search is deliberately NOT windowed (a real
    # band may sit arbitrarily far past an unmatched run), and capping
    # its input here would silently reintroduce that regression.
    cap = own_words * _BRACKET_SLACK
    whole_table_words = table_words
    if bracketed and anchor is None and len(table_words) > cap:
        issues.append(
            Issue(
                code="ALIGN_BRACKET_UNANCHORED_TRUNCATED",
                severity="warning",
                node_id=group[0].node_id,
                path=group[0].path,
                message=f"{len(table_words)} words bracketed, "
                f"{own_words} own words, k={_BRACKET_SLACK}",
                details={
                    "picked": len(table_words),
                    "own_words": own_words,
                    "k": _BRACKET_SLACK,
                },
            )
        )
        whole_table_words = table_words[:cap]
    if bracketed:
        search = _Search(remaining=_TABLE_SEARCH_BUDGET)
        try:
            assignment = _resolve_block(whole_table_words, cells, columns, search)
        except _SearchExhausted:
            assignment = None
        grid_rejected = False
    else:
        assignment = located_assignment
    if assignment is not None and not _rows_in_document_order(cells, assignment):
        # An assignment that contradicts the grid is EVIDENCE that this
        # table cannot be told apart — two cells of the same text in one
        # column, read in the other order. Falling back to bands there
        # would hand the first band the wrong one of the two and box it,
        # which is the very thing the check exists to refuse. Bands are
        # for the absence of a resolution, not for a rejected one.
        grid_rejected = True
        assignment = {}
    elif assignment is None:
        # the table as a whole did not resolve: keep the bands that do
        band_cursor = [0]
        assignment = _assign_by_bands(
            table_words,
            cells,
            columns,
            issues,
            group[0].node_id,
            group[0].path,
            resume=band_cursor,
        )
        if band_cursor[0] > 0:
            if band_cursor[0] < len(picked):
                search_start = max(search_start, picked[band_cursor[0]])
            elif picked:
                search_start = max(search_start, picked[-1] + 1)
    unresolved = {
        member
        for cell in cells
        if cell.key not in assignment
        for member in cell.members
    }
    if unresolved and not grid_rejected:
        resume = max(
            resume,
            _match_table_row_fallback(
                cells, group, words_per_paragraph, pdf_words,
                search_start, search_end,
                issues, consumed, assignment, index, next_texts,
            ),
        )
        unresolved = {
            member
            for cell in cells
            if cell.key not in assignment
            for member in cell.members
        }
    if unresolved:
        stream = [w.text for w in table_words]
        for index, (paragraph, words) in enumerate(zip(group, words_per_paragraph)):
            if words and index in unresolved:
                raw_words = [
                    paragraph.text[char_start:char_end]
                    for _key, char_start, char_end in words
                ]
                issues.append(_mismatch_issue(paragraph, raw_words, stream))
    if not assignment:
        return resume
    # `matched_ids` (0f, "cm"): only the words a cell's paragraphs
    # actually spelled — `cluster[start:end]` per paragraph span, NOT
    # the whole `assignment[cell.key]` cluster. A bracketed cluster can
    # be far bigger than what verified it (the single-cell terminal case
    # of `_resolve_block` accepts the WHOLE candidate list once
    # `_cell_spans` finds a prefix match inside it), so marking the
    # whole cluster consumed — the previous behaviour — let one
    # over-eager bracket bury words a LATER group still needed (class A).
    # Orphan-reporting the cluster's leading/trailing leftover is
    # likewise skipped when `bracketed`: those words are not spent here,
    # so the sweep at the end of `_match_paragraphs` sees and reports
    # them once, on the whole document, rather than here on a region
    # that turned out to be too wide.
    matched_ids: set[int] = set()
    matched_chars = 0
    for cell in cells:
        cluster = assignment.get(cell.key)
        if cluster is None:
            continue
        spans = _cell_spans(cluster, cell.paragraphs)
        assert spans is not None  # the assignment was chosen by this very check
        cursor = 0
        for j, (start, end) in zip(cell.members, spans):
            if start > cursor and not bracketed:
                issues.append(_orphan_issue(cluster[cursor:start], group[j].path))
            _consume(
                group[j], words_per_paragraph[j], cluster[start:end],
                global_owned_ids=global_owned_ids,
            )
            for w in cluster[start:end]:
                wid = id(w)
                if wid not in matched_ids:
                    matched_ids.add(wid)
                    matched_chars += len(w.key)
            cursor = end
        if cursor < len(cluster) and not bracketed:
            # the words after the cell's last paragraph. They are inside
            # the region and therefore already marked consumed, so the
            # sweep at the end of the pass cannot see them: reported here
            # or reported nowhere. Same word and same geometry as the
            # ones before the first paragraph, which were always
            # reported — a glyph that is neither matched nor reported
            # would contradict this module's contract and would make
            # "the orphan count reaches zero" mean less than it says.
            issues.append(_orphan_issue(cluster[cursor:], group[cell.members[-1]].path))
    if bracketed:
        for i in picked:
            if id(pdf_words[i]) in matched_ids:
                consumed[i] = 1
        if matched_chars > own_chars * _BRACKET_SLACK:
            # Character count, not token count: a split or joined PDF
            # token changes the token count without changing the text,
            # and the slack is about how much TEXT one table consumed,
            # not how many poppler words spelled it. The unanchored
            # whole-table attempt above is bounded, and the band
            # fallback verifies each row independently by exact
            # character match, so this residual check is mainly a named
            # backstop — it still fires for the ANCHORED case, which is
            # not bounded (design §0b): a genuine finding (a class-A
            # variant WITH an anchor), left for its own fix rather than
            # truncated here.
            issues.append(
                Issue(
                    code="ALIGN_GROUP_OVERCONSUMED_ANCHORED"
                    if anchor is not None
                    else "ALIGN_GROUP_OVERCONSUMED",
                    severity="warning",
                    node_id=group[0].node_id,
                    path=group[0].path,
                    message=f"{matched_chars} characters consumed, "
                    f"{own_chars} own characters, k={_BRACKET_SLACK}",
                    details={
                        "consumed_chars": matched_chars,
                        "own_chars": own_chars,
                        "k": _BRACKET_SLACK,
                    },
                )
            )
    return resume


def _assign_by_bands(
    words: list[_StreamWord],
    cells: list[_BlockCell],
    columns: tuple[int, int],
    issues: list[Issue],
    node_id: str,
    path: str,
    resume: list[int] | None = None,
) -> dict[int, list[_StreamWord]]:
    """Resolve one closed row band at a time, keeping the bands that
    verify and leaving the rest to be reported.

    Scoping the assignment to the whole table is what fixed the merged
    cell, and it is also all-or-nothing: one band that cannot be
    resolved refuses every other band with it. On the synthetic cohort
    that never showed, because its tables are small and tidy. On the
    real corpus it is the common case — a 101-cell hearing list lost
    every cell because ONE header cell (`Esito`) is not in the PDF's
    text layer at all, where the row-at-a-time matcher this replaced
    lost only that row (186 of 204 nodes against 0).

    So the table is tried whole first, and this runs only when that
    fails. It is not a weaker check: every cell is still verified by
    exact character equality inside its own band, and a band whose top
    lies above the band before it is skipped rather than assigned, the
    same rule :func:`_rows_in_document_order` applies to the whole. What
    it gives up is the cross-band evidence a merged cell needs — which
    is why it is the fallback and not the method.
    """
    assignment: dict[int, list[_StreamWord]] = {}
    remaining = cells
    cursor = 0
    previous_top: tuple[int, int] | None = None
    # how many bands have failed since the last one that resolved: a
    # band that resolves puts a floor under the next one, a band that
    # fails leaves none
    missing = 0
    # One aggregate budget for this table's ENTIRE band pass (T10):
    # every band's per-band search charges it as its parent, so no
    # matter how many bands the table has, the pass spends at most
    # this much work in total. Per-band budgets alone leave the table
    # unbounded — see :data:`_TABLE_BANDS_BUDGET`. When it runs out the
    # band loop stops for this table (see `except _SearchExhausted`
    # below) and the rows it never reached fall through to
    # `_match_table`'s residual path.
    table_bands = _Search(remaining=_TABLE_BANDS_BUDGET, budget=_TABLE_BANDS_BUDGET)
    while remaining:
        band: tuple[int, list[_BlockCell]] | None = None
        for last in sorted({cell.row_start for cell in remaining}):
            head = [cell for cell in remaining if cell.row_start <= last]
            if not any(cell.row_end > last + 1 for cell in head):
                band = (last, head)
                break
        if band is None:
            break  # every remaining band is merged into a later one
        last, head = band
        remaining = [cell for cell in remaining if cell.row_start > last]
        # A closed band can hold several rows, because a merged cell
        # binds them together. If it does not resolve, its rows are
        # tried one at a time — which is what the row-at-a-time matcher
        # did, and all it could ever do: a merged cell's text then
        # belongs to a row whose glyphs do not hold it, so that row is
        # lost and the others are not. Worth having only because the
        # alternative here is losing all of them.
        groups = [head]
        rows = sorted({cell.row_start for cell in head})
        if len(rows) > 1:
            groups.append(None)  # marker: retry row by row
        for attempt in groups:
            batches = (
                [attempt]
                if attempt is not None
                else [[c for c in head if c.row_start == row] for row in rows]
            )
            progress = False
            for batch in batches:
                target = "".join(w[0] for c in batch for p in c.paragraphs for w in p)
                at = cursor
                segment = resolved = None
                # A fresh, per-batch budget (design §0c): sharing the
                # whole table's `search` here let one long table's early
                # bands spend it all, aborting every band after ~135
                # rows with a `return`. A budget scoped to this one band
                # means an exhausted band refuses on its own — see
                # `except _SearchExhausted` below — without starving the
                # rest of the table. T10: it charges the whole band
                # pass's aggregate budget as its parent, so the table
                # itself is bounded — and the work it pays is the work
                # the finders and `_resolve_block` actually do, not a
                # flat slice of the word list.
                batch_search = _Search(
                    remaining=_BAND_SEARCH_BUDGET,
                    budget=_BAND_SEARCH_BUDGET,
                    parent=table_bands,
                )
                tolerant_placement = False
                try:
                    # exact first; the tolerant fallback (0f, "tolfb")
                    # runs only once the exact finder cannot place this
                    # band at all — never for the uniqueness recheck
                    # below, which stays exact throughout.
                    for finder in (_find_row_segment, _tolerant_row_segment):
                        at = cursor
                        while True:
                            # T10: the finder charges the words it
                            # actually visits. The flat `len(words) - at`
                            # pre-charge it replaced priced words the
                            # scan never saw and left the worst-case
                            # scan free — the budget measured a tenth of
                            # the work on the corpus shape that
                            # triggered T10.
                            segment = finder(target, words, at, window=None, search=batch_search)
                            if segment is None:
                                break
                            resolved = _resolve_block(
                                words[segment[0] : segment[1]], batch, columns, batch_search
                            )
                            if resolved is not None:
                                break
                            at = segment[0] + 1  # a match that does not resolve is not it
                        if resolved is not None:
                            tolerant_placement = finder is _tolerant_row_segment
                            break

                    if (
                        resolved is not None
                        and tolerant_placement
                        and _tolerant_window_hides_a_later_row(
                            remaining, words, cursor, segment[1]
                        )
                    ):
                        # not this band's glyphs tolerating some noise —
                        # the NEXT row's glyphs, tolerated as the noise.
                        resolved = None

                    if resolved is not None and missing:
                        # No floor under this band: the band before it
                        # failed, so its words were never found and this
                        # one is free to match ABOVE them — row 1 took
                        # row 0's glyph, row 1's own glyph was reported
                        # orphan, and every mismatch sat on row 0, so
                        # from outside it read exactly like the intended
                        # "lost only that row". Where there is no floor,
                        # place the band only where nothing else could
                        # be it, which is the rule the rest of this
                        # module already uses for a position it cannot
                        # trust. "Could be" means resolves: a multiset
                        # match whose cells do not spell out is not a
                        # candidate. Measured on 3 000 tables of the
                        # shape this fallback exists for, wrong boxes
                        # 49 -> 7 while the boxes placed go 15 042 ->
                        # 14 997; requiring room for the failed band
                        # instead was tried and made it 102, because
                        # skipping the early candidate takes a late one.
                        #
                        # That 0.3 % is the GENERATOR's cost and it
                        # understates the corpus by two orders: its
                        # tables have 2-5 bands, so a band rarely has a
                        # failure above it, while a real 100-row table
                        # puts every band after the first failure under
                        # this rule. Over ten real documents it costs
                        # 184 nodes on four (`04783e31` 315 -> 219,
                        # `51668809` 110 -> 70) and wins 211 on one
                        # (`59897b38` 2 065 -> 2 276): net +27, and no
                        # grid violations either way. The trade is
                        # deliberate — a missing box shows up as lower
                        # coverage and leaves `page_evidence`
                        # unresolved, a wrong box trains against a lie.
                        elsewhere = segment[0] + 1
                        while True:
                            # charged the same way as the main finder —
                            # the recheck is part of the band's work
                            other = _find_row_segment(
                                target, words, elsewhere, window=None, search=batch_search
                            )
                            if other is None:
                                break
                            if (
                                _resolve_block(
                                    words[other[0] : other[1]], batch, columns, batch_search
                                )
                                is not None
                            ):
                                resolved = None  # two places could be it
                                break
                            elsewhere = other[0] + 1
                except _SearchExhausted:
                    if table_bands.remaining < 0:
                        # the table's aggregate budget is spent, not
                        # just this band's: the band pass is stopped for
                        # this table, ONE issue named for it, and the
                        # rows it never reached fall through to
                        # `_match_table`'s residual path — the same
                        # path any unresolved row already takes. A
                        # per-band issue here would emit one per
                        # remaining band for the same event. The
                        # classification is sound because
                        # `_Search.spend` charges the parent before the
                        # child: the first raise with the parent spent
                        # can only be the parent's, and vice versa.
                        issues.append(
                            Issue(
                                code="ALIGN_TABLE_BANDS_BUDGET_EXHAUSTED",
                                severity="warning",
                                node_id=node_id,
                                path=path,
                                message=f"row {last} table band search exhausted "
                                f"({table_bands.budget} words of work)",
                                details={"row": last, "budget": table_bands.budget},
                            )
                        )
                        if resume is not None:
                            resume[0] = cursor
                        return assignment
                    # this band's own budget is spent, not the table's:
                    # report it and try the next band with no floor
                    # under it, the same as any other unresolved band —
                    # never abandon the rest of the table for it.
                    issues.append(
                        Issue(
                            code="ALIGN_BAND_BUDGET_EXHAUSTED",
                            severity="warning",
                            node_id=node_id,
                            path=path,
                            message=f"row {last} band search exhausted "
                            f"({_BAND_SEARCH_BUDGET} words of work)",
                            details={"row": last},
                        )
                    )
                    missing += 1
                    continue
                if segment is None or resolved is None:
                    missing += 1
                    continue
                if not _rows_in_document_order(batch, resolved):
                    missing += 1
                    continue
                top = min(
                    (word.page_index, word.bbox_ppm[1])
                    for cluster in resolved.values()
                    for word in cluster
                )
                if previous_top is not None and top < previous_top:
                    missing += 1
                    continue  # this band renders above the one before it
                assignment.update(resolved)
                cursor = segment[1]
                previous_top = top
                missing = 0
                progress = True
            if progress:
                break
    if resume is not None:
        resume[0] = cursor
    return assignment


def _rows_in_document_order(
    cells: list[_BlockCell], assignment: dict[int, list[_StreamWord]]
) -> bool:
    """Does the assignment put the table's row bands down the page in
    the order the AST states?

    The one thing exact character equality cannot check. Two cells of
    one column holding the SAME text are indistinguishable to it, so
    where the stream lists them in the other order — which is what a
    scrambled read order does, and this whole module exists because read
    order is not to be trusted — the row peel takes the first and the
    boxes come out exchanged, silently. `_split_columns` refuses that
    shape on the column axis, because two assignments verify and it
    counts them; the row axis cannot count, because the repaint above
    needs the FIRST of two identical bands rather than a refusal.

    So the grid is used instead, on both of its axes: a row band is
    rendered below the bands before it, and within a row a lower
    `grid_column` is rendered left of a higher one — on the same page or
    an earlier one. It is a check on geometry against AST topology, with
    no distance in it: cells may start at the same height or the same x,
    but a later one may not start above or left of an earlier one.
    `Yes` / `N/A` / `-` in one column is common, and it is the shape the
    row axis closes; the column axis closes a row whose cells were
    matched on different pages in the wrong order — excusing a
    backwards step only where the cell BEFORE it is merged across rows,
    which is the one that a page break can carry forward.

    UNTESTED CORNER of that excuse: it is granted on the earlier cell,
    so a misplaced cell standing immediately after a carried merged one
    inherits it, and its own inversion is never tested. Building that
    shape was attempted and could not be made to fire — the aligner
    picked the right glyph and the stray became orphan text — so it is
    recorded as a corner rather than a defect. The excuse is still
    strictly narrower than the one it replaced, which granted itself to
    any row holding a merged cell at all.
    """
    tops: dict[int, tuple[int, int]] = {}
    lefts: dict[tuple[int, int], tuple[tuple[int, int], bool]] = {}
    for cell in cells:
        cluster = assignment.get(cell.key, [])
        spans = _cell_spans(cluster, cell.paragraphs)
        if spans is None:
            continue
        # the words the cell's paragraphs actually matched, not the whole
        # cluster: on the anchor path a cluster also holds the words
        # before and after them, and one stray above a band dragged the
        # band's top up and refused an assignment that was entirely
        # right. Measured over 2 000 generated tables, the cluster form
        # refused 23 correct assignments to catch 3 wrong ones.
        # UNPROVEN in its detail: taking the FIRST span's words instead
        # of all of them leaves the suite green and differs on 0 of
        # 4 000 generated tables (mutated 2026-09-08), because a cell's
        # paragraphs are matched in stream order and run down the page.
        # The distinction from the CLUSTER, below, is proven.
        matched = [word for start, end in spans for word in cluster[start:end]]
        if not matched:
            continue
        top = min((word.page_index, word.bbox_ppm[1]) for word in matched)
        tops[cell.row_start] = min(tops.get(cell.row_start, top), top)
        seat = (cell.row_start, cell.column_start)
        left = min((word.page_index, word.bbox_ppm[0]) for word in matched)
        carried = cell.row_end - cell.row_start > 1
        if seat not in lefts or left < lefts[seat][0]:
            lefts[seat] = (left, carried)
    ordered = [tops[row] for row in sorted(tops)]
    if not all(a <= b for a, b in zip(ordered, ordered[1:])):
        return False
    # and across a row, a lower grid_column lies left of a higher one —
    # on the same page or an earlier one. The row axis alone let a row's
    # LAST cell take a box on an EARLIER page than its first, which is
    # how both of the real corpus's first two directly-visible wrong
    # boxes look: `fa8b1100` has columns 0-6 on page 1 and column 7 on
    # page 0, `67924f32` column 0 on page 57 and column 1 on page 56.
    for row in {row for row, _column in lefts}:
        across = [
            lefts[(row, column)]
            for column in sorted(c for r, c in lefts if r == row)
        ]
        for (earlier, earlier_carried), (later, _later_carried) in zip(across, across[1:]):
            if later >= earlier:
                continue
            # A backwards step is excusable only when the EARLIER cell is
            # the merged one. That cell straddles the page break, and
            # Word carries its text to the continuation page while the
            # cells beside it stay on the first — later page, earlier
            # column, and correct. When the merged cell is the one on
            # the EARLIER page it was not carried anywhere: it is
            # misplaced, and excusing it by merging alone let every
            # `fa8b1100`-class inversion through whenever the offending
            # cell happened to span rows, with the other cells boxed as
            # if all were well.
            if not earlier_carried:
                return False
    return True


def _key_stream(pdf_words: list[PdfWord]) -> list[_StreamWord]:
    """The pairing-key view of the PDF word stream (0g).

    One pass over the document's words: :func:`normalize_for_matching`
    is a fixed point, so matching a key against a key would change
    nothing, and the raw text rides along for diagnostics and for
    counting key-assisted matches.
    """
    stream: list[_StreamWord] = []
    for word in pdf_words:
        key, key_to_raw = normalize_for_matching(word.text)
        stream.append(
            _StreamWord(
                page_index=word.page_index,
                text=word.text,
                key=key,
                key_to_raw=tuple(key_to_raw),
                bbox_ppm=word.bbox_ppm,
            )
        )
    return stream


def _keyed_words(paragraph: _Paragraph) -> list[tuple[str, int, int]]:
    """The paragraph's words as pairing keys (0g).

    ``normalize_for_matching(apply_caps(word, start, caps_spans))`` —
    caps applied in the raw text first, then invisible characters
    dropped and ligatures expanded — with the offsets kept in the
    ORIGINAL text's index space, so fragments and run spans never move.
    """
    keyed: list[tuple[str, int, int]] = []
    for word, start, end in _words_with_offsets(paragraph.text):
        key, _ = normalize_for_matching(apply_caps(word, start, paragraph.caps_spans))
        keyed.append((key, start, end))
    if paragraph.phantoms:
        keyed = _splice_phantoms(keyed, paragraph.phantoms)
    return keyed


def _splice_phantoms(
    keyed: list[tuple[str, int, int]], phantoms: list[tuple[int, str]]
) -> list[tuple[str, int, int]]:
    """`keyed` with each phantom `(offset, key)` inserted as a zero-length
    `(key, offset, offset)` entry at its seam -- the same shape as the
    residual pass's note-marker tier 1. A seam inside one keyed word (runs
    joined without whitespace around the field) splits that word there;
    this assumes the key is as long as the raw text up to the cut, which
    holds outside a ligature or invisible character on the seam."""
    pieces = list(keyed)
    for offset, phantom in sorted(phantoms, key=lambda item: item[0]):
        if not phantom:
            continue
        out: list[tuple[str, int, int]] = []
        inserted = False
        for key, start, end in pieces:
            if not inserted and start < offset < end:
                cut = offset - start
                out += [
                    (key[:cut], start, offset),
                    (phantom, offset, offset),
                    (key[cut:], offset, end),
                ]
                inserted = True
            elif not inserted and (start > offset or (start == offset and end > offset)):
                out += [(phantom, offset, offset), (key, start, end)]
                inserted = True
            else:
                out.append((key, start, end))
        if not inserted:
            out.append((phantom, offset, offset))
        pieces = out
    return pieces


def _textnorm_issue(count: int, samples: list[tuple[str, str]]) -> Issue:
    """The document-level record that pairing needed the key (0g).

    One per document: the count is what the sidecar and the gates read
    (it survives the ``issue_details`` projection, which keeps only
    code / node_id / message), the samples are for a human.
    """
    return Issue(
        code="OBSERVATION_ALIGNMENT_TEXTNORM_MATCH",
        severity="info",
        node_id=None,
        path="/",
        message=f"{count} word matches required text normalization "
        "(invisible character, ligature or caps)",
        details={"count": count, "samples": samples},
    )


def _key_assisted_stats(
    walker: _Walker, stories: StoryAlignment
) -> tuple[int, list[tuple[str, str]]]:
    """How many word matches held only because of the key (0g), with up
    to five (raw_ast, raw_pdf) samples for the issue."""
    count = 0
    samples: list[tuple[str, str]] = []
    for paragraph in walker.paragraphs:
        for word in paragraph.matched_words:
            if not word.key_assisted:
                continue
            count += 1
            if len(samples) < 5 and word.raw_pdf is not None:
                samples.append(
                    (paragraph.text[word.char_start : word.char_end], word.raw_pdf)
                )
    count += stories.key_assisted_count
    for sample in stories.key_assisted_samples:
        if len(samples) >= 5:
            break
        samples.append(sample)
    return count, samples


def _match_paragraphs(
    paragraphs: list[_Paragraph],
    pdf_words: list[_StreamWord],
    issues: list[Issue],
    *,
    page_break_split: bool = False,
    tab_segments: bool = False,
) -> None:
    """Match paragraphs against the PDF stream in reading order,
    mutating ``paragraphs`` in place and appending issues. Paragraphs
    of one table form a single group — the renderer interleaves the
    wrapped lines of a row's cells, and a cell merged across rows
    interleaves the rows — and everything else matches sequentially.
    A nested table is its own group, because the walker gives a
    paragraph its INNERMOST table.

    ``pdf_words`` is the keyed stream (:func:`_key_stream`); the AST
    side is keyed per paragraph by :func:`_keyed_words`, so matching
    runs on the key while every stored offset stays in the raw text
    (0g)."""
    groups: list[list[_Paragraph]] = []
    for paragraph in paragraphs:
        if (
            groups
            and paragraph.table_id is not None
            and groups[-1][0].table_id == paragraph.table_id
        ):
            groups[-1].append(paragraph)
        else:
            groups.append([paragraph])

    index = _StreamIndex(pdf_words)
    words_per_group = [[_keyed_words(p) for p in group] for group in groups]
    # A table located between its anchors takes words that are not one
    # run of the stream, so "everything before ``pos``" no longer says
    # what has been accounted for and each word has to say it itself.
    consumed = bytearray(len(pdf_words))
    global_owned_ids: set[int] = set()
    pos = 0
    for group_index, group in enumerate(groups):
        words_per_paragraph = words_per_group[group_index]
        if not any(words_per_paragraph):
            continue
        next_texts = next(
            (
                [w[0] for words in wpp for w in words]
                for wpp in words_per_group[group_index + 1 :]
                if any(wpp)
            ),
            [],
        )
        match = _match_sequential if len(group) == 1 else _match_table
        extra: dict[str, Any] = {}
        if len(group) == 1 and page_break_split:
            extra["page_break_split"] = True
        if len(group) == 1 and tab_segments:
            extra["tab_segments"] = True
        pos = match(
            group, words_per_paragraph, pdf_words, pos, issues, consumed, index, next_texts,
            global_owned_ids=global_owned_ids, **extra,
        )
        while pos < len(pdf_words) and consumed[pos]:
            pos += 1
    leftover = [word for i, word in enumerate(pdf_words) if not consumed[i]]
    if leftover:
        issues.append(_orphan_issue(leftover, "/"))


def align_document(
    ast: dict[str, Any],
    pdf_words: list[PdfWord],
    *,
    section_spans: dict[int, tuple[int, int]] | None = None,
    story_mode: str = "dec033",
    frames: bool = True,
    residual: bool = True,
    residual_strict: bool = False,
    min_key_chars: int | None = None,
    policy: AlignerPolicy | None = None,
) -> AlignmentResult:
    """Align the AST's visible text to ``pdf_words`` (document order,
    all pages concatenated). Returns fragments in node preorder, plus
    typed issues; text-node counts cover paragraphs and runs that carry
    at least one word.

    Body text is matched in one sequential pass; header/footer/note
    stories are NOT in reading order and are matched per page by
    :mod:`unflatten.dataset.observations.stories`, and `text_frame`
    content is matched per page by
    :mod:`unflatten.dataset.observations.frames` — both sets of
    fragments are merged in here so every consumer of the sidecar sees
    one set of observations.

    ``section_spans``, when given, is passed through to `align_stories`
    unchanged instead of being recomputed from ``pdf_words`` — a hook
    for `reading_order.pick_reading_order` to pin story placement across
    its resync rounds (T1 x T9c); every other caller passes nothing and
    keeps today's per-call recomputation.

    ``story_mode`` and ``frames`` (T1 x T9c round 4) are the same kind
    of hook, for the reading-order PICK DECISION specifically:
    ``story_mode="legacy_first_match"`` reproduces the pre-DEC-033
    ``align_stories`` behaviour (first match only, no keep-all, no
    section span — see `stories.align_stories`), and ``frames=False``
    skips the frame pass entirely, reproducing pre-DEC-033
    `align_document` byte-for-byte. `pick_reading_order` uses both ONLY
    to choose which pages switch; the alignment it actually returns is
    always built with the defaults (``"dec033"``, frames on). Every
    other caller passes neither and is unaffected.

    ``residual`` (T1 x T3, §3.2/§4.2) runs the residual pass after the
    body, story and frame passes, over words no paragraph spelled: it
    claims unmatched table cells and `note_ref` paragraphs whose exact
    text is a unique run among them. It never changes an existing
    match, so it plays no part in the reading-order PICK DECISION —
    `reading_order._legacy_align` passes ``residual=False`` for exactly
    that reason, the same way it passes ``frames=False``.
    ``residual_strict`` raises `AlignmentInvariantViolation` on a
    single-owner conflict instead of counting and skipping it; unit
    tests only (production always counts and skips).

    ``policy`` (`AlignerPolicy`, e.g. `V16_ALIGNER_POLICY`) opts into the
    v16 alignment rules; ``None`` is today's alignment, byte for byte.
    """
    from unflatten.dataset.observations.frames import FrameAlignment
    from unflatten.dataset.observations.frames import align_frames as _align_frames
    from unflatten.dataset.observations.residual import MIN_KEY_CHARS, residual_pass
    from unflatten.dataset.observations.stories import _full_page_spans, align_stories

    if min_key_chars is None:
        min_key_chars = MIN_KEY_CHARS

    walker = _Walker()
    walker.walk(ast)
    issues: list[Issue] = []
    stream = _key_stream(pdf_words)
    _match_paragraphs(
        walker.paragraphs, stream, issues,
        page_break_split=policy is not None and policy.page_break_split,
        tab_segments=policy is not None and policy.body_tab_segments,
    )

    story_kwargs: dict[str, Any] = {}
    if policy is not None and policy.story_field_results:
        story_kwargs["field_results"] = True
    if policy is not None and policy.story_segment_claims:
        story_kwargs["segment_claims"] = True
    if policy is not None and policy.story_frames:
        story_kwargs["story_frames"] = True
    stories = align_stories(
        ast, stream, section_spans=section_spans, story_mode=story_mode,
        body_claimed=frozenset(id(w) for p in walker.paragraphs for w in p.spelled_words),
        **story_kwargs,
    )
    issues.extend(stories.issues)

    frame_kwargs: dict[str, Any] = {}
    if policy is not None and policy.story_frames and story_mode != "legacy_first_match":
        frame_kwargs = {
            "story_pages": stories.story_pages,
            "field_result": stories.field_result,
            "segment_claims": policy.story_segment_claims,
        }
    frames_result = (
        _align_frames(ast, stream, walker, **frame_kwargs)
        if frames
        else FrameAlignment(fragments=[], issues=[], matched_text_nodes=0, total_text_nodes=0)
    )
    issues.extend(frames_result.issues)

    repeat_fragments: list[Fragment] = []
    repeat_word_ids: set[int] = set()
    if policy is not None and policy.repeat_header_rows and story_mode != "legacy_first_match":
        from unflatten.dataset.observations.stories import align_repeated_header_rows

        repeat_fragments, repeat_word_ids = align_repeated_header_rows(ast, stream, walker)

    residual_result = None
    if residual:
        excluded_boxes_by_page: dict[int, list[tuple[int, int, int, int]]] = {}
        for placed in (*stories.fragments, *frames_result.fragments, *repeat_fragments):
            excluded_boxes_by_page.setdefault(placed.page_index, []).append(placed.bbox_ppm)
        resolved_section_spans = section_spans
        if resolved_section_spans is None:
            resolved_section_spans = _full_page_spans(
                _body_section_spans(walker), sorted({word.page_index for word in stream})
            )
        residual_result = residual_pass(
            walker,
            stream,
            excluded_boxes_by_page,
            _margins_by_page(ast, resolved_section_spans),
            strict=residual_strict,
            min_key_chars=min_key_chars,
            policy=policy,
        )

    claimed_word_ids = (
        {id(word) for word in residual_result.claimed_words} if residual_result else set()
    )
    claimed_word_ids |= repeat_word_ids
    if policy is not None and policy.body_tab_segments:
        # A16 claims a label an earlier paragraph skipped as orphan text
        claimed_word_ids |= {id(w) for p in walker.paragraphs for w in p.spelled_words}
    issues = _finalize_orphan_issues(issues, claimed_word_ids)

    ordered, matched_text_nodes, total_text_nodes = collect_fragments(walker)
    ordered.sort(key=lambda item: item[0])
    fragments = [fragment for _, group in ordered for fragment in group]
    fragments.extend(stories.fragments)
    fragments.extend(frames_result.fragments)
    fragments.extend(repeat_fragments)

    key_count, key_samples = _key_assisted_stats(walker, stories)
    if key_count:
        issues.append(_textnorm_issue(key_count, key_samples))
    return AlignmentResult(
        fragments=fragments,
        issues=issues,
        matched_text_nodes=(
            matched_text_nodes + stories.matched_text_nodes + frames_result.matched_text_nodes
        ),
        total_text_nodes=(
            total_text_nodes + stories.total_text_nodes + frames_result.total_text_nodes
        ),
        frames_total=frames_result.frames_total,
        frames_placed=frames_result.frames_placed,
        frame_runs_total=frames_result.frame_runs_total,
        frame_runs_observed=frames_result.frame_runs_observed,
        frame_text_scalars=frames_result.frame_text_scalars,
        residual_claimed=len(residual_result.claimed) if residual_result else 0,
        residual_refused_ambiguous=residual_result.refused_ambiguous if residual_result else 0,
        residual_refused_contested=residual_result.refused_contested if residual_result else 0,
        residual_not_found=residual_result.not_found if residual_result else 0,
        residual_rounds=residual_result.rounds if residual_result else 0,
        residual_invariant_skips=residual_result.invariant_skips if residual_result else 0,
        note_marker_matches=residual_result.note_marker_matches if residual_result else 0,
        field_fragments=list(stories.field_fragments),
    )


def collect_fragments(
    walker: _Walker,
) -> tuple[list[tuple[int, list[Fragment]]], int, int]:
    """Observation fragments for everything `walker` collected.

    Returns ``(fragments keyed by walker preorder, matched, total)`` for
    paragraphs, their runs and their container ancestors. Shared with
    the per-page story pass, which builds the same shapes from a
    different match strategy.
    """
    ordered: list[tuple[int, list[Fragment]]] = []
    matched_text_nodes = 0
    total_text_nodes = 0

    for paragraph in walker.paragraphs:
        has_words = _has_glyphs(paragraph.text)
        if has_words:
            total_text_nodes += 1
            if paragraph.matched:
                matched_text_nodes += 1
        ordered.append(
            (
                paragraph.preorder,
                _fragments(paragraph.node_id, paragraph.matched_words, len(paragraph.text)),
            )
        )
        for run_id, run_preorder, run_start, run_end in paragraph.run_spans:
            run_len = run_end - run_start
            run_text = paragraph.text[run_start:run_end]
            if not _has_glyphs(run_text):
                continue  # empty, whitespace-only, or DEL/C0-only run: no glyphs
            total_text_nodes += 1
            run_words = [
                _MatchedWord(
                    page_index=w.page_index,
                    bbox_ppm=w.bbox_ppm,
                    char_start=max(w.char_start, run_start) - run_start,
                    char_end=min(w.char_end, run_end) - run_start,
                )
                for w in paragraph.matched_words
                if w.char_start < run_end and w.char_end > run_start
            ]
            if run_words:
                matched_text_nodes += 1
            ordered.append((run_preorder, _fragments(run_id, run_words, run_len)))

    for container in walker.containers.values():
        words: list[_MatchedWord] = []
        offset = 0
        for paragraph in container.paragraphs:
            for w in paragraph.matched_words:
                words.append(
                    _MatchedWord(
                        page_index=w.page_index,
                        bbox_ppm=w.bbox_ppm,
                        char_start=w.char_start + offset,
                        char_end=w.char_end + offset,
                    )
                )
            offset += len(paragraph.text) + 1  # "\n" separator between paragraphs
        total_len = max(offset - 1, 0)
        ordered.append((container.preorder, _fragments(container.node_id, words, total_len)))

    return ordered, matched_text_nodes, total_text_nodes
