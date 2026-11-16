"""Per-page alignment of header / footer / note stories (DEC-003 §2).

A story is the one part of the document whose glyphs are NOT in the
document's reading order. A header has ink on *every* page of its
section, a footer's glyphs sit after the body text of the page it
belongs to, and a footnote body sits at the foot of the page its
reference is on — so the single sequential pass in
:mod:`unflatten.dataset.observations.aligner`, which consumes one
monotone word stream, cannot place any of them. Before this module they
were simply never walked: a header node reached the Observation Sidecar
with no observations at all.

That was not a cosmetic gap. An unobserved node carrying ink is
``unresolved`` page evidence (`fragments.envelope`), whose
``placement_origin`` is ``imputed_for_render`` — and RFC005 §9.2 blocks
quality training on ``imputed_for_render`` supervision. A header would
therefore have been present in the target and excluded from the
training it exists to enable.

**The rule (DEC-003 §2, reading (a), superseded for section stories by
DEC-033 §3): a header/footer is owned by its section and takes
supervision from EVERY PAGE OF ITS OWN SECTION that actually renders
it — each page's placement is a self-contained chain (`chain_scope:
"page"`, see `sidecar._semantic_issues`), because under DEC-033 every
page is a standalone document and repeating ink on page 2 is that
page's own target, not a duplicate of page 1's. A note body still
renders once, so NOTE_KEYS keep the original first-match,
single-placement rule, except the final v16 exact continuation fallback
for a uniquely spelled multi-page note (RFC003 §24.3).**

"every page OF ITS OWN SECTION" is load-bearing: two sections whose
headers happen to render identical text (a common case — the same
"Chapter N" style repeated verbatim) would otherwise each match the
OTHER section's pages too, giving one rendered page two header
observations for one visible header. A section's own page span is
derived from where its non-story body content is actually observed
(`_section_body_page_spans`), then widened so every page belongs to a
section (`_full_page_spans`); a story searches only inside its own
section's span plus the spans of later sections that inherit it
(`_heir_sections`, OOXML header/footer inheritance).
`titlePg` narrows this further: `kind == "first"` is bound to the
section's own first observed page only, so it does not also claim the
`default` header's pages (a page-span-only rule would not separate the
two, since both are the same section). `kind == "even"`/`"default"`
have no reliable page-parity signal in the AST (no absolute physical
page numbering is stored), so they stay bound to the whole section
span — the "obvious rule" the design calls for is not derivable there.

Matching is the same exact word-sequence match the body uses, run
against ONE page's words at a time. Orphan words are not reported here —
on a page-scoped search every body word is an orphan by construction,
which says nothing — but an alignment miss is: a story that matches no
page yields ``OBSERVATION_STORY_UNPLACED`` and no observations, so it
falls back to ``unresolved`` rather than to a fabricated page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import unflatten.dataset.observations.aligner as _aligner
from unflatten.ast.issues import Issue
from unflatten.dataset.observations.aligner import (
    FieldFragment,
    Fragment,
    _consume,
    _find_char_span,
    _keyed_words,
    _Paragraph,
    _span_at,
    _StreamIndex,
    _StreamWord,
    _union_bbox,
    _Walker,
    _words_with_offsets,
    collect_fragments,
)
from unflatten.dataset.observations.textnorm import normalize_for_matching

# the note-story root keys, from the module that defines the canonical
# preorder, so the two orders cannot drift apart
from unflatten.dataset.targets.events import NOTE_KEYS, preorder_nodes

#: Section-level story keys, in the canonical preorder the canonicalizer
#: numbers ids in (`canonicalizer._SECTION_CHILD_KEYS`).
SECTION_STORY_KEYS: tuple[str, ...] = ("headers", "footers")

_SECTION_PATH_RE = re.compile(r"^/sections/(\d+)/")


@dataclass
class StoryAlignment:
    """Per-page story alignment result, in `AlignmentResult` shape."""

    fragments: list[Fragment]
    issues: list[Issue]
    matched_text_nodes: int
    total_text_nodes: int
    #: word matches that held only because of the pairing key (0g)
    key_assisted_count: int = 0
    #: (raw_ast, raw_pdf) samples of those matches
    key_assisted_samples: list[tuple[str, str]] = field(default_factory=list)
    #: section stories placed under the first-match fallback because
    #: their own section has no observed body page span (DEC-033 §3
    #: addendum)
    stories_without_section_span: int = 0
    #: field-only header/footer paragraphs observed by their computed
    #: result (policy `story_field_results`)
    field_fragments: list[FieldFragment] = field(default_factory=list)
    #: section story path -> the pages it may render on
    #: (`_story_candidate_pages`), and the per-page field result function;
    #: filled under a policy only, for the frames inside those stories (A13)
    story_pages: dict[str, list[int]] = field(default_factory=dict)
    field_result: Any = None


@dataclass(frozen=True)
class _Story:
    node: dict[str, Any]
    path: str


def stories_of(ast: dict[str, Any]) -> list[_Story]:
    """Every story body of `ast`, in canonical preorder."""
    stories: list[_Story] = []
    for section_index, section in enumerate(ast.get("sections", [])):
        for key in SECTION_STORY_KEYS:
            for index, node in enumerate(section.get(key) or []):
                stories.append(_Story(node, f"/sections/{section_index}/{key}/{index}"))
    for key in NOTE_KEYS:
        for index, node in enumerate(ast.get(key) or []):
            stories.append(_Story(node, f"/{key}/{index}"))
    return stories


def _is_section_story(story: _Story) -> bool:
    """Is `story` a header/footer (page-scoped, DEC-033 §3), as opposed
    to a note body (`NOTE_KEYS`, document-scoped, first match only)?"""
    return story.path.startswith("/sections/")


def _marker_texts_by_note_id(ast: dict[str, Any]) -> dict[str, str]:
    """``note_id -> marker_text`` for every `note_ref` in `ast` (§4.3,
    N-S): a note body carries no text for its own marker (only the
    reference elsewhere does), so this is a generic whole-document walk
    rather than a lookup local to the note — a `note_ref` can sit in
    body text, a header, footer, table cell or frame."""
    marker_texts: dict[str, str] = {}
    stack: list[Any] = [ast]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("type") == "note_ref":
                note_id = node.get("note_id")
                marker_text = node.get("marker_text")
                if note_id and marker_text is not None:
                    marker_texts.setdefault(note_id, marker_text)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return marker_texts


def _story_section_index(story: _Story) -> int | None:
    """The section index a section story (header/footer) belongs to, or
    `None` for a note body, which has no section in its path."""
    m = _SECTION_PATH_RE.match(story.path)
    return int(m.group(1)) if m else None


def _section_body_page_spans(
    ast: dict[str, Any], pdf_words: list[_StreamWord]
) -> dict[int, tuple[int, int]]:
    """The inclusive ``(min_page, max_page)`` each section's own body
    (non-story) content is observed on, keyed by section index.

    A header/footer may only be placed on pages that belong to its OWN
    section (DEC-033 §3 addendum): two sections whose stories render
    identical text would otherwise each match the other's pages too,
    giving one rendered page two header observations. This runs the
    same body-only walk+match `align_document` already performs, a
    second time — a duplicate cost accepted for this fix's scope, since
    stories are matched from a per-page slice and have no other way to
    see where the body itself landed.

    A section absent from the returned mapping has no observed body at
    all (nothing matched); callers keep the original first-match rule
    for its stories rather than guessing a span (design R6 addendum).
    """
    walker = _Walker()
    walker.walk(ast)
    _aligner._match_paragraphs(walker.paragraphs, pdf_words, [])
    spans: dict[int, tuple[int, int]] = {}
    for paragraph in walker.paragraphs:
        m = _SECTION_PATH_RE.match(paragraph.path)
        if not m:
            continue
        section_index = int(m.group(1))
        for word in paragraph.matched_words:
            lo, hi = spans.get(section_index, (word.page_index, word.page_index))
            spans[section_index] = (min(lo, word.page_index), max(hi, word.page_index))
    return spans


def section_story_id_sets(ast: dict[str, Any]) -> dict[str, frozenset[str]]:
    """Map each header/footer's root node id to the set of node ids it
    owns (itself plus its paragraphs, runs and containers).

    AST-only — no PDF matching — so callers can attribute *observed*
    fragments (by ``node_id``) back to the story that owns them without
    re-running alignment. Used by `sidecar.build_observation_sidecar`
    for the `chain_scope` tag and the per-page placement counters.
    """
    result: dict[str, frozenset[str]] = {}
    for story in stories_of(ast):
        if not _is_section_story(story):
            continue
        walker = _walk(story)
        ids = {story.node["id"]}
        for paragraph in walker.paragraphs:
            ids.add(paragraph.node_id)
            ids.update(run_id for run_id, *_ in paragraph.run_spans)
        ids.update(walker.containers)
        result[story.node["id"]] = frozenset(ids)
    return result


def page_scoped_node_ids(ast: dict[str, Any], *, story_frames: bool = False) -> frozenset[str]:
    """Every node id owned by a header/footer — the nodes whose
    observations carry `chain_scope: "page"` (DEC-033 §3). Note bodies
    stay document-scoped, so they are not in this set.

    ``story_frames`` (policy `v16-align.v1`, A13): a `text_frame` inside a
    header/footer and its whole subtree are placed on every page of the
    story (`frames.align_frames`), so they are page-scoped too."""
    ids: set[str] = set()
    for owned in section_story_id_sets(ast).values():
        ids.update(owned)
    if story_frames:
        from unflatten.dataset.observations.frames import _story_path, frames_of
        from unflatten.dataset.observations.images import _subtree_ids

        for frame in frames_of(ast):
            if _story_path(frame.path) is not None:
                ids.update(_subtree_ids(frame.node))
    # Image observations do not count as story text placements, but they
    # repeat per rendered page and therefore need independent page chains.
    from unflatten.dataset.observations.images import header_footer_image_node_ids

    ids.update(header_footer_image_node_ids(ast))
    return frozenset(ids)


def all_story_node_ids(ast: dict[str, Any]) -> frozenset[str]:
    """Every node id owned by ANY story — header, footer, or note body —
    as opposed to `page_scoped_node_ids`, which covers only the
    page-scoped ones (headers/footers, not notes).

    AST-only, same convention as `section_story_id_sets`. Used by
    `reading_order.pick_reading_order` to tell body fragments from
    story fragments when deciding whether to switch a page's reading
    order: story fragment counts are not a signal of THIS page's body
    quality (T1 x T9c finding, 2026-09), so the switch decision must
    ignore them.
    """
    ids: set[str] = set()
    for story in stories_of(ast):
        walker = _walk(story)
        ids.add(story.node["id"])
        for paragraph in walker.paragraphs:
            ids.add(paragraph.node_id)
            ids.update(run_id for run_id, *_ in paragraph.run_spans)
        ids.update(walker.containers)
    return frozenset(ids)


def _full_page_spans(
    body_spans: dict[int, tuple[int, int]], all_pages: list[int]
) -> dict[int, tuple[int, int]]:
    """Widen the body spans of `_section_body_page_spans` so every page of
    the document belongs to a section (DEC-033 F1, prod diagnosis D2):
    pages before the first matched page go to the first spanned section,
    a gap goes to the preceding section, the tail to the last one. With
    no matched body at all, the whole document is section 0's span. A
    section with no matched body still gets no span of its own."""
    if not all_pages:
        return dict(body_spans)
    if not body_spans:
        return {0: (all_pages[0], all_pages[-1])}
    ordered = sorted(body_spans)
    spans: dict[int, tuple[int, int]] = {}
    for position, section_index in enumerate(ordered):
        lo, hi = body_spans[section_index]
        if position == 0:
            lo = min(lo, all_pages[0])
        if position + 1 < len(ordered):
            hi = max(hi, body_spans[ordered[position + 1]][0] - 1)
        else:
            hi = max(hi, all_pages[-1])
        spans[section_index] = (lo, hi)
    return spans


def _heir_sections(ast: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Story path -> the later sections that display it by inheritance.

    OOXML: a section with no header/footer reference of a type and kind
    shows the previous section's one, and the renderer writes references
    only for a section's own stories (`document_builder._build_sect_pr`),
    so inheritance is what gets rendered (prod diagnosis D1). `first` is
    not inherited here: the renderer sets `titlePg` only on a section
    with its own `first` story, so a section without one never shows a
    first-page variant at all."""
    heirs: dict[str, list[int]] = {}
    current: dict[tuple[str, str], list[str]] = {}
    for section_index, section in enumerate(ast.get("sections", [])):
        own: dict[tuple[str, str], list[str]] = {}
        for key in SECTION_STORY_KEYS:
            for index, node in enumerate(section.get(key) or []):
                if node.get("kind") == "first":
                    continue
                own.setdefault((node.get("type"), node.get("kind")), []).append(
                    f"/sections/{section_index}/{key}/{index}"
                )
        for type_kind, paths in current.items():
            if type_kind not in own:
                for path in paths:
                    heirs.setdefault(path, []).append(section_index)
        current.update(own)
    return {path: tuple(sections) for path, sections in heirs.items()}


def _story_ranges(
    story: _Story, spans: dict[int, tuple[int, int]], heirs: dict[str, tuple[int, ...]]
) -> list[tuple[int, int]]:
    """The page spans of `story`'s own section and of every section that
    inherits it (`_heir_sections`), in section order."""
    sections = (_story_section_index(story), *heirs.get(story.path, ()))
    return [spans[section] for section in sections if section in spans]


def _story_candidate_pages(
    story: _Story,
    spans: dict[int, tuple[int, int]],
    all_pages: list[int],
    heirs: dict[str, tuple[int, ...]],
) -> tuple[list[int], bool]:
    """``(candidate_pages, has_span)`` -- the pages `_place_story`
    searches for `story`'s words, before matching. Shared by
    `_place_story` and `story_page_coverage` so the coverage
    denominator's notion of "expected" never drifts from what placement
    actually searches (`story` must be a section story;
    `_is_section_story` is the caller's job). ``spans`` are the
    full-page spans (`_full_page_spans`). ``has_span=False`` means
    neither the story's section nor any section inheriting it has a
    span, the `_place_story` fallback case (module docstring, design R6
    addendum)."""
    ranges = _story_ranges(story, spans, heirs)
    if not ranges:
        return all_pages, False
    if story.node.get("kind") == "first":
        return [p for p in all_pages if any(p == lo for lo, _hi in ranges)], True
    # "default"/"even": no page-parity signal is stored in the AST, so
    # these stay bound to the whole span (module docstring).
    return [p for p in all_pages if any(lo <= p <= hi for lo, hi in ranges)], True


def _story_has_words(story: _Story) -> bool:
    """Does `story` have any matchable glyphs at all (as opposed to an
    image-only header or an empty footer)?"""
    walker = _walk(story)
    return any(_words_with_offsets(p.text) for p in walker.paragraphs)


def story_page_coverage(
    ast: dict[str, Any], observations_by_page: dict[int, list[dict[str, Any]]]
) -> tuple[int, int, int, int]:
    """``(story_pages_covered, story_pages_total, story_pages_
    glyphless_only, story_pages_not_applicable)`` -- the store-gate axis
    10 coverage counters (design §2, `spec-page-standalone-targets`,
    corrected per the prod DEC-033 rebuild diagnosis).

    A page enters the denominator (`story_pages_total`) only when at
    least one of its section's header/footer stories is BOTH
    non-glyphless AND *expected* on this specific page -- expectation
    uses the exact kind/span rule `_place_story` applies
    (`_story_candidate_pages`), so the denominator never drifts from
    what placement actually searches for. `story_pages_covered` counts
    those pages where one of the expected stories was actually placed
    there (whichever story rendered -- when a `default` and a `first`
    story of the SAME type both apply, either being observed covers the
    page). The `titlePg` override is the one case `_story_candidate_
    pages` alone does not settle: a `default` story is not expected on
    the section's own first page when a `first` story of the same type
    (`header`/`footer`) exists there, since Word replaces it outright --
    `_place_story`'s own span rule does not need to know this (an
    unmatched `default` attempt there is simply harmless), but the
    coverage denominator does, or a glyphless `first` (blank title-page
    header) would count that page as a real miss instead of a
    legitimate absence.

    Spans and inheritance are the ones placement uses
    (`_full_page_spans`, `_heir_sections`), so the `titlePg` override
    also applies to an inherited `default` on a page where the page's
    own section has a `first` story.

    A page in the scope of some story but where NONE apply (a
    `first`-only header on a later page) is `story_pages_not_applicable`. A page whose only
    applicable stories are all glyphless is `story_pages_glyphless_
    only`. Both are informational, no threshold (R5-style). A page with
    in no story's scope (no span, or no story of its own or inherited) is
    not counted in any of the four."""
    walker = _Walker()
    walker.walk(ast)
    section_by_node: dict[str, int] = {}
    for paragraph in walker.paragraphs:
        m = _SECTION_PATH_RE.match(paragraph.path)
        if m:
            section_by_node[paragraph.node_id] = int(m.group(1))

    all_pages = sorted(observations_by_page)
    section_pages: dict[int, set[int]] = {}
    for page_index, observations in observations_by_page.items():
        for obs in observations:
            section_index = section_by_node.get(obs["node_id"])
            if section_index is not None:
                section_pages.setdefault(section_index, set()).add(page_index)
    spans = _full_page_spans(
        {section: (min(pages), max(pages)) for section, pages in section_pages.items()},
        all_pages,
    )
    heirs = _heir_sections(ast)
    story_owned_ids = section_story_id_sets(ast)

    # per story: the pages in its sections' scope, and the pages it is
    # expected on (`_story_candidate_pages`)
    scoped: list[tuple[_Story, set[int], set[int], bool]] = []
    first_pages: set[tuple[str, int]] = set()
    for story in stories_of(ast):
        if not _is_section_story(story):
            continue
        ranges = _story_ranges(story, spans, heirs)
        if not ranges:
            continue
        scope = {p for p in all_pages if any(lo <= p <= hi for lo, hi in ranges)}
        candidates, _has_span = _story_candidate_pages(story, spans, all_pages, heirs)
        scoped.append((story, scope, set(candidates), _story_has_words(story)))
        if story.node.get("kind") == "first":
            first_pages.update((story.node.get("type"), p) for p in candidates)

    covered = 0
    total = 0
    glyphless_only = 0
    not_applicable = 0
    for page_index, observations in observations_by_page.items():
        observed_ids = {obs["node_id"] for obs in observations}
        in_scope = False
        applicable_any = False
        expected_stories: list[_Story] = []
        for story, scope, candidates, has_words in scoped:
            if page_index not in scope:
                continue
            in_scope = True
            if (
                story.node.get("kind") == "default"
                and (story.node.get("type"), page_index) in first_pages
            ):
                # titlePg override (docstring): a `first` story of the
                # same type replaces `default` on that page, whether or
                # not it is glyphless.
                continue
            if page_index not in candidates:
                continue
            applicable_any = True
            if has_words:
                expected_stories.append(story)
        if not in_scope:
            continue

        if expected_stories:
            total += 1
            if any(
                observed_ids & story_owned_ids.get(story.node["id"], frozenset())
                for story in expected_stories
            ):
                covered += 1
        elif applicable_any:
            glyphless_only += 1
        else:
            not_applicable += 1
    return covered, total, glyphless_only, not_applicable


def glyphless_section_story_count(ast: dict[str, Any]) -> tuple[int, int]:
    """``(glyphless, total)`` among header/footer stories — a glyphless
    story (image-only header, empty footer) has no matchable words, so
    it is never placed and stays on one page by `_resolve_pages`
    (design R5): informative, not an alignment failure."""
    glyphless = 0
    total = 0
    for story in stories_of(ast):
        if not _is_section_story(story):
            continue
        total += 1
        walker = _walk(story)
        if not any(_words_with_offsets(p.text) for p in walker.paragraphs):
            glyphless += 1
    return glyphless, total


# -- computed field results (policy `story_field_results`, A10/A11) ------------


def _format_page_number(number: int, fmt: str) -> str | None:
    """`number` in a `section.page_numbering.format`; `None` when the
    format cannot spell it (roman/alpha of 0 or less)."""
    if fmt == "decimal":
        return str(number)
    if number < 1:
        return None
    if fmt in ("lower_roman", "upper_roman"):
        numerals = (
            (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
            (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
        )  # fmt: skip
        out, rest = "", number
        for value, digits in numerals:
            while rest >= value:
                out, rest = out + digits, rest - value
        return out if fmt == "lower_roman" else out.upper()
    if fmt in ("lower_alpha", "upper_alpha"):
        # Word's scheme: a..z, then aa..zz, aaa..
        letter = chr(ord("a") + (number - 1) % 26) * ((number - 1) // 26 + 1)
        return letter if fmt == "lower_alpha" else letter.upper()
    return None


def page_labels(
    ast: dict[str, Any], spans: dict[int, tuple[int, int]], n_pages: int
) -> dict[int, str]:
    """The PAGE field result of every page: numbering continues across
    pages and restarts at a section's first page when its
    `page_numbering.start` is set (`null` = continue, the schema's
    reading), in that section's format. ``spans`` are the full page spans
    (`_full_page_spans`); a page no span covers continues in decimal."""
    sections = ast.get("sections", [])
    section_of: dict[int, int] = {}
    for section_index, (lo, hi) in spans.items():
        for page in range(lo, hi + 1):
            section_of.setdefault(page, section_index)
    # a section restarts on the first page it OWNS: two spans may share a
    # page (a stray body match; m2a-v7 7c0491465d6c section 2 (1, 20) beside
    # section 1 (1, 1)), and the page goes to the earlier section above
    first_of: dict[int, int] = {}
    for page in sorted(section_of):
        first_of.setdefault(section_of[page], page)
    labels: dict[int, str] = {}
    number = 0
    for page in range(n_pages):
        section_index = section_of.get(page)
        numbering = (
            (sections[section_index].get("page_numbering") or {})
            if section_index is not None and section_index < len(sections)
            else {}
        )
        first = first_of.get(section_index) if section_index is not None else None
        if page == first and numbering.get("start") is not None:
            number = numbering["start"]
        else:
            number += 1
        label = _format_page_number(number, numbering.get("format") or "decimal")
        if label is not None:
            labels[page] = label
    return labels


#: a field-only header/footer line must sit in this share of the page height
#: from its edge (top for a header, bottom for a footer)
_FIELD_EDGE_ZONE_PPM = 250_000


def _edge_line(words: list[_StreamWord], footer: bool) -> list[_StreamWord]:
    """The page's last (footer) or first (header) visual line: the words
    whose vertical extent overlaps the lowest / highest word's, in stream
    order -- empty when that line lies outside `_FIELD_EDGE_ZONE_PPM`.
    A running page number renders on the page's edge line by construction
    (the footer below the body, the header above it); the page's margin
    does not bound it, because LibreOffice grows a footer upward into the
    page when the bottom margin is smaller than the footer."""
    if not words:
        return []
    center = [(w.bbox_ppm[1] + w.bbox_ppm[3]) // 2 for w in words]
    edge = words[max(range(len(words)), key=center.__getitem__)] if footer else words[
        min(range(len(words)), key=center.__getitem__)
    ]
    y0, y1 = edge.bbox_ppm[1], edge.bbox_ppm[3]
    edge_center = (y0 + y1) // 2
    if (footer and edge_center < 1_000_000 - _FIELD_EDGE_ZONE_PPM) or (
        not footer and edge_center > _FIELD_EDGE_ZONE_PPM
    ):
        return []
    return [w for w in words if w.bbox_ppm[1] < y1 and w.bbox_ppm[3] > y0]


def _place_field_only_story(
    story: _Story,
    words_by_page: dict[int, list[_StreamWord]],
    candidate_pages: list[int],
    field_result: Any,
) -> list[FieldFragment]:
    """A header/footer whose only visible content is fields (19 of the 22
    [pagenum] pages of the G-C trial): on each candidate page, every
    field-only paragraph (exactly one field, no glyphs) must be spelled by
    exactly one run of words on the page's edge line (`_edge_line`: the last
    line for a footer, the first for a header) -- its computed result, never
    "any number" -- or the page is not claimed."""
    paragraphs = _walk(story).paragraphs
    if any(_words_with_offsets(p.text) for p in paragraphs):
        # a story with glyphs is placed by its text (`_place_story`); its
        # field-only paragraphs follow the story root (envelope rule 0)
        return []
    targets = [p for p in paragraphs if p.field_gaps]
    if not targets or any(len(p.field_gaps) != 1 for p in targets):
        return []
    footer = story.node.get("type") == "footer"
    placed: list[FieldFragment] = []
    for page in candidate_pages:
        in_band = _edge_line(list(words_by_page.get(page, ())), footer)
        page_placed: list[FieldFragment] = []
        for paragraph in targets:
            _offset, field_type, node_id = paragraph.field_gaps[0]
            result = field_result(field_type, page)
            key = normalize_for_matching(result)[0] if result else ""
            spans = [
                span for i in range(len(in_band)) if key and (span := _span_at(key, in_band, i))
            ]
            if len(spans) != 1:
                page_placed = []
                break
            words = in_band[spans[0][0] : spans[0][1]]
            page_placed.append(
                FieldFragment(node_id, page, _union_bbox([w.bbox_ppm for w in words]))
            )
        placed.extend(page_placed)
    return placed


def _story_frames(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Text frames among a story's blocks (inside its lists and tables too)."""
    out: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("type") == "text_frame":
            out.append(block)
        elif block.get("type") == "list":
            for item in block.get("items", []):
                out += _story_frames(item.get("children", []))
        elif block.get("type") == "table":
            for row in block.get("rows", []):
                for cell in row.get("cells", []):
                    out += _story_frames(cell.get("children", []))
    return out


def _place_field_only_frames(
    story: _Story,
    words_by_page: dict[int, list[_StreamWord]],
    candidate_pages: list[int],
    field_result: Any,
) -> list[FieldFragment]:
    """A13 for a glyphless frame in a header/footer whose content is one
    field (a page number boxed in a footer frame: 41 pages of m2a-v7
    67a9f00c604a): `_place_field_only_story`'s edge-line rule, applied to
    the frame as if it were the story."""
    out: list[FieldFragment] = []
    for number, frame in enumerate(_story_frames(story.node.get("children", []))):
        as_story = _Story({**story.node, "children": frame.get("children", [])},
                          f"{story.path}/frames/{number}")  # fmt: skip
        out += _place_field_only_story(as_story, words_by_page, candidate_pages, field_result)
    return out


def _unambiguous_field_claims(
    claims: list[tuple[_Story, list[FieldFragment]]],
) -> list[FieldFragment]:
    """Drop a field claim another story also makes on the same page box --
    except that a single `first` story wins it (titlePg shows only the
    first-page variant on the section's first page)."""
    by_box: dict[tuple[int, tuple[int, int, int, int]], list[tuple[int, str, FieldFragment]]] = {}
    for number, (story, fragments) in enumerate(claims):
        for fragment in fragments:
            by_box.setdefault((fragment.page_index, fragment.bbox_ppm), []).append(
                (number, story.node.get("kind") or "default", fragment)
            )
    out: list[FieldFragment] = []
    for owners in by_box.values():
        stories = {number for number, _kind, _f in owners}
        firsts = {number for number, kind, _f in owners if kind == "first"}
        if len(stories) == 1:
            out.extend(f for _n, _k, f in owners)
        elif len(firsts) == 1:
            out.extend(f for number, _k, f in owners if number in firsts)
    return sorted(out, key=lambda f: (f.page_index, f.node_id, f.bbox_ppm))


def align_stories(
    ast: dict[str, Any],
    pdf_words: list[_StreamWord],
    *,
    section_spans: dict[int, tuple[int, int]] | None = None,
    story_mode: str = "dec033",
    field_results: bool = False,
    segment_claims: bool = False,
    story_frames: bool = False,
    body_claimed: frozenset[int] = frozenset(),
) -> StoryAlignment:
    """Observe every header/footer on every page whose words spell it
    (DEC-033 §3); a note body still keeps the first-match rule.

    ``pdf_words`` is the whole document's word stream in pairing-key
    space — the same keyed stream
    :func:`~unflatten.dataset.observations.aligner.align_document`
    builds once and shares with the body pass; it is sliced per page
    here.

    ``section_spans``, when given, is used AS-IS instead of being
    recomputed from ``pdf_words`` (T1 x T9c, reading_order pin): a page
    whose own body match differs between rounds of
    `reading_order.pick_reading_order`'s resync would otherwise move
    every section's span between rounds, and with it every story's
    placement — never converging on some real documents. The default
    (``None``) recomputes the spans from ``pdf_words`` exactly as
    before, so every caller other than `pick_reading_order` is
    unaffected.

    ``story_mode="legacy_first_match"`` (T1 x T9c round 4) forces every
    story, section-scoped or not, onto the pre-DEC-033 first-match rule
    — no keep-all, no section span. `reading_order.pick_reading_order`
    uses it ONLY to decide the switched page set (§3's keep-all
    placement made the per-page story count a property of the whole
    document, not of one page's own text, which the pick cannot
    reason about); the final returned alignment always uses the default
    ``"dec033"`` mode.

    ``field_results`` (policy `v16-align.v1`, A10/A11; ``dec033`` mode
    only): a header/footer `field` is matched as its computed result on
    each page tried (`page_labels`, NUMPAGES = the page count), and a
    field-only header/footer is placed by `_place_field_only_story`.

    ``segment_claims`` (policy `v16-align.v1`, A14; ``dec033`` mode only):
    a section story paragraph no ordered match spells on a page is claimed
    segment by segment (`_claim_segments`).

    ``story_frames`` (policy `v16-align.v1`, A13, with ``field_results``):
    a glyphless text frame inside a header/footer whose one content is a
    field is placed like a field-only story (`_place_field_only_frames`).

    ``body_claimed`` reserves actual body word identities for the note
    continuation fallback. Existing one-page placements keep their rules.
    """
    words_by_page: dict[int, list[_StreamWord]] = {}
    for word in pdf_words:
        words_by_page.setdefault(word.page_index, []).append(word)
    heirs: dict[str, tuple[int, ...]] = {}
    marker_texts: dict[str, str] = {}
    if story_mode != "legacy_first_match":
        if section_spans is None:
            section_spans = _section_body_page_spans(ast, pdf_words)
        section_spans = _full_page_spans(section_spans, sorted(words_by_page))
        heirs = _heir_sections(ast)
        # N-S (§4.3): the final alignment only, never the reading-order
        # pick decision — this walk is skipped in `legacy_first_match`.
        marker_texts = _marker_texts_by_note_id(ast)

    fragments: list[Fragment] = []
    issues: list[Issue] = []
    matched_nodes = 0
    total_nodes = 0
    key_count = 0
    key_samples: list[tuple[str, str]] = []
    without_span = 0
    field_result = None
    field_claims: list[tuple[_Story, list[FieldFragment]]] = []
    segment_claims = segment_claims and story_mode != "legacy_first_match"
    story_pages: dict[str, list[int]] = {}
    pending_notes: list[tuple[_Story, list[Issue]]] = []
    if field_results and story_mode != "legacy_first_match" and words_by_page:
        n_pages = max(words_by_page) + 1
        labels = page_labels(ast, section_spans, n_pages)

        def field_result(field_type: str, page: int) -> str | None:
            if field_type == "page":
                return labels.get(page)
            if field_type == "num_pages":
                return str(n_pages)
            return None

    for story in stories_of(ast):
        if (field_result is not None or segment_claims) and _is_section_story(story):
            candidate_pages, has_span = _story_candidate_pages(
                story, section_spans, sorted(words_by_page), heirs
            )
            if has_span:
                # without a span a story falls back to one placement
                # (`_place_story`), and so do the frames inside it
                story_pages[story.path] = candidate_pages
        if field_result is not None and _is_section_story(story):
            field_claims.append(
                (
                    story,
                    _place_field_only_story(story, words_by_page, candidate_pages, field_result)
                    + (
                        _place_field_only_frames(
                            story, words_by_page, candidate_pages, field_result
                        )
                        if story_frames
                        else []
                    ),
                )
            )
        (
            placed,
            story_issues,
            matched,
            total,
            story_key_count,
            story_key_samples,
            no_span,
        ) = _place_story(
            story, words_by_page, section_spans, heirs,
            story_mode=story_mode, marker_texts=marker_texts,
            field_result=field_result,
            segment_claims=segment_claims,
        )
        fragments.extend(placed)
        issues.extend(story_issues)
        if segment_claims and not placed and story_issues and not _is_section_story(story):
            pending_notes.append((story, story_issues))
        matched_nodes += matched
        total_nodes += total
        key_count += story_key_count
        key_samples.extend(story_key_samples)
        without_span += no_span

    # The frame pass runs later and has its own ownership. Until a joint
    # claim API exists, a document with frames is outside this fallback's
    # scope. Existing one-page note/story/frame alignment is unchanged.
    if pending_notes and not any(n["type"] == "text_frame" for n in preorder_nodes(ast)):
        boxes_by_page: dict[int, set[tuple[int, int, int, int]]] = {}
        for fragment in fragments:
            boxes_by_page.setdefault(fragment.page_index, set()).add(fragment.bbox_ppm)
        reserved = body_claimed | frozenset(
            id(w) for page, words in words_by_page.items() for w in words
            if any(
                w.bbox_ppm[0] < box[2] and box[0] < w.bbox_ppm[2]
                and w.bbox_ppm[1] < box[3] and box[1] < w.bbox_ppm[3]
                for box in boxes_by_page.get(page, ())
            )
        )
        proposals = []
        for story, old_issues in pending_notes:
            proposal = _retry_note_continuation(
                story, words_by_page, marker_texts.get(story.node["id"]), reserved
            )
            if proposal is not None:
                proposals.append((old_issues, *proposal))
        for number, (old_issues, continued, claimed) in enumerate(proposals):
            if any(
                claimed & other for other_number, (_issues, _result, other) in enumerate(proposals)
                if number != other_number
            ):
                continue  # competing notes BOTH stay unplaced, irrespective of AST order
            placed, _issues, matched, _total, counted, samples, _no_span = continued
            fragments.extend(placed)
            issues = [i for i in issues if all(i is not old for old in old_issues)]
            matched_nodes += matched
            key_count += counted
            key_samples.extend(samples)
    return StoryAlignment(
        fragments=fragments,
        issues=issues,
        matched_text_nodes=matched_nodes,
        total_text_nodes=total_nodes,
        key_assisted_count=key_count,
        key_assisted_samples=key_samples,
        stories_without_section_span=without_span,
        field_fragments=_unambiguous_field_claims(field_claims),
        story_pages=story_pages,
        field_result=field_result,
    )


def _match_with_label_prefix(
    paragraph: _Paragraph, page_words: list[_StreamWord], marker_key: str
) -> bool:
    """Try matching `paragraph` with `marker_key` as a phantom prefix
    (§4.3, N-S): the note-area label LibreOffice paints, which no AST
    node carries text for — a zero-length ``(marker_key, 0, 0))``
    phantom entry, exactly like the residual pass's `note_ref` tier 1
    (`observations.residual`)."""
    keyed = _keyed_words(paragraph)
    if not keyed:
        return False
    plain_key = "".join(key for key, _start, _end in keyed)
    target = marker_key + plain_key
    ast_words = [(marker_key, 0, 0), *keyed]
    index = _StreamIndex(page_words)
    for start in index.candidate_starts(target, 0):
        span = _span_at(target, page_words, start)
        if span is not None:
            _consume(paragraph, ast_words, page_words[span[0] : span[1]])
            return True
    return False


def _retry_with_note_label(
    story: _Story,
    words_by_page: dict[int, list[_StreamWord]],
    candidate_pages: list[int],
    marker_text: str | None,
) -> tuple[list[Fragment], list[Issue], int, int, int, list[tuple[str, str]], int] | None:
    """The second loop of §4.3: only reached after the whole exact loop
    found nothing on any page. Tries every candidate page again, this
    time with the note's own marker as a phantom prefix of its first
    glyph-bearing paragraph; every other paragraph still needs its
    ordinary exact (or gap) match."""
    if not marker_text:
        return None
    marker_key, _key_to_raw = normalize_for_matching(marker_text)
    if not marker_key:
        return None
    for page_index in candidate_pages:
        attempt = _walk(story)
        with_words = [p for p in attempt.paragraphs if _words_with_offsets(p.text)]
        if not with_words:
            continue
        first, rest = with_words[0], with_words[1:]
        page_words = words_by_page[page_index]
        if not _match_with_label_prefix(first, page_words, marker_key):
            continue
        if not all(
            paragraph.matched or _match_across_gaps(paragraph, page_words, beyond_window=True)
            for paragraph in rest
        ):
            continue
        ordered, matched, total = collect_fragments(attempt)
        ordered.sort(key=lambda item: item[0])
        key_count, key_samples = _key_assisted(attempt)
        fragments = [fragment for _, group in ordered for fragment in group]
        return fragments, [], matched, total, key_count, key_samples, 0
    return None


def _place_story(
    story: _Story,
    words_by_page: dict[int, list[_StreamWord]],
    section_spans: dict[int, tuple[int, int]] | None,
    heirs: dict[str, tuple[int, ...]],
    *,
    story_mode: str = "dec033",
    marker_texts: dict[str, str] | None = None,
    field_result: Any = None,
    segment_claims: bool = False,
) -> tuple[list[Fragment], list[Issue], int, int, int, list[tuple[str, str]], int]:
    """Place ``story`` on the page(s) its words spell.

    A header/footer (`_is_section_story`) keeps every matching page
    WITHIN ITS OWN SECTION and the sections that inherit it — DEC-033 §3, each page is a standalone
    document and repeat ink is that page's own target, but a section
    boundary still bounds which pages can be "its own" (the addendum
    this function implements; see the module docstring). A note body
    keeps the original rule: the first matching page only, because it
    renders once.

    If neither the story's section nor any heir has a span
    (`section_spans` has no entry for them), there is no span to bound the search by —
    this falls back to the original first-match rule and is counted in
    the returned ``stories_without_section_span`` tally rather than
    silently searching every page.

    Every page produces its own self-contained fragment chain
    (`_fragments` groups by page, so a one-page attempt always yields
    ``fragment_index`` starting at 0 with both continuation flags
    False) — `chain_scope` is stamped on the observation later, in
    `sidecar.build_observation_sidecar`, from `page_scoped_node_ids`.
    """
    walker = _walk(story)
    with_words = [p for p in walker.paragraphs if _words_with_offsets(p.text)]
    if not with_words:
        # A story with no matchable glyphs (an image-only header, an
        # empty footer). There is nothing to search for; its placement
        # stays for `page_evidence` to characterize.
        return [], [], 0, 0, 0, [], 0

    keep_all = story_mode != "legacy_first_match" and _is_section_story(story)
    without_span = 0
    candidate_pages = sorted(words_by_page)
    if keep_all:
        candidate_pages, has_span = _story_candidate_pages(
            story, section_spans, candidate_pages, heirs
        )
        if not has_span:
            keep_all = False
            without_span = 1
            candidate_pages = sorted(words_by_page)

    fragments: list[Fragment] = []
    matched_total = 0
    total_total = 0
    key_count_total = 0
    key_samples_total: list[tuple[str, str]] = []

    for page_index in candidate_pages:
        attempt = _walk(story)
        page_words = words_by_page[page_index]
        if field_result is not None:
            _set_field_phantoms(attempt, field_result, page_index)
        _aligner._match_paragraphs(attempt.paragraphs, page_words, [])
        for paragraph in attempt.paragraphs:
            paragraph.phantoms = []  # the gap fallback sees the plain paragraph
        if not all(
            paragraph.matched
            or _match_across_gaps(
                paragraph, page_words, beyond_window=story_mode != "legacy_first_match"
            )
            for paragraph in attempt.paragraphs
            if _words_with_offsets(paragraph.text)
        ) and not (
            segment_claims
            and _is_section_story(story)
            and _claim_segments(attempt.paragraphs, page_words, field_result, page_index)
        ):
            continue
        ordered, matched, total = collect_fragments(attempt)
        ordered.sort(key=lambda item: item[0])
        story_key_count, story_key_samples = _key_assisted(attempt)
        fragments.extend(fragment for _, group in ordered for fragment in group)
        matched_total += matched
        total_total += total
        key_count_total += story_key_count
        key_samples_total.extend(story_key_samples)
        if not keep_all:
            return (
                fragments,
                [],
                matched_total,
                total_total,
                key_count_total,
                key_samples_total,
                without_span,
            )

    if fragments:
        return (
            fragments,
            [],
            matched_total,
            total_total,
            key_count_total,
            key_samples_total,
            without_span,
        )

    if not _is_section_story(story) and story_mode != "legacy_first_match":
        # N-S (§4.3): a second loop, only after the WHOLE exact loop
        # above found nothing on any page — a note that matches exactly
        # somewhere is never pre-empted by an earlier page's label
        # match. Disabled in `legacy_first_match` (the reading-order
        # pick decision, `reading_order._legacy_align`) on purpose:
        # `_decide_switched` must see today's note placements only.
        marker_text = (marker_texts or {}).get(story.node["id"])
        retried = _retry_with_note_label(story, words_by_page, candidate_pages, marker_text)
        if retried is not None:
            return retried

    _unplaced, _matched, total = collect_fragments(walker)
    return (
        [],
        [
            Issue(
                code="OBSERVATION_STORY_UNPLACED",
                severity="error",
                node_id=story.node["id"],
                path=story.path,
                message=(
                    f"{story.node['type']} text was not found on any rendered page; "
                    "no page is claimed for it (DEC-003 §2)."
                ),
                details={"pages_searched": candidate_pages},
            )
        ],
        0,
        total,
        0,
        [],
        without_span,
    )


def _note_prefix_matches(
    keyed: list[tuple[str, int, int]],
    words: list[_StreamWord],
    body_claimed: frozenset[int],
) -> list[tuple[int, list[_StreamWord]]]:
    """Maximal exact prefixes ending at BOTH AST and PDF word boundaries.

    No missing glyph is skipped within a page. Body-owned words cannot
    supply note ink. A repeated prefix remains separate evidence, so the
    caller refuses competing complete chains instead of picking one.
    """
    ends: dict[int, int] = {}
    target = ""
    for number, (key, _start, _end) in enumerate(keyed, 1):
        target += key
        ends[len(target)] = number
    matches = []
    for start, word in enumerate(words):
        if id(word) in body_claimed or not target.startswith(word.key):
            continue
        cursor = 0
        best = None
        for end in range(start, len(words)):
            word = words[end]
            if id(word) in body_claimed or not target.startswith(word.key, cursor):
                break
            cursor += len(word.key)
            if cursor in ends:
                best = (ends[cursor], words[start : end + 1])
            if cursor == len(target):
                break
        if best is not None:
            matches.append(best)
    return matches


def _retry_note_continuation(
    story: _Story,
    words_by_page: dict[int, list[_StreamWord]],
    marker_text: str | None,
    body_claimed: frozenset[int],
) -> tuple[
    tuple[list[Fragment], list[Issue], int, int, int, list[tuple[str, str]], int],
    frozenset[int],
] | None:
    """RFC003 §24.3: one uniquely spelled note paragraph across adjacent pages.

    Only a complete, gap-free chain is admitted, after existing one-page
    placement failed. Unsupported multi-paragraph/table/field stories stay
    unplaced. Each state retains at most two paths: two is already enough
    to prove ambiguity, even when those paths share their final suffix.
    """
    attempt = _walk(story)
    paragraphs = [p for p in attempt.paragraphs if _words_with_offsets(p.text)]
    if len(paragraphs) != 1:
        return None
    paragraph = paragraphs[0]
    if paragraph.table_id is not None or paragraph.glyph_gaps:
        return None
    keyed = _keyed_words(paragraph)
    marker_key = normalize_for_matching(marker_text)[0] if marker_text else ""
    variants = [keyed]
    if marker_key:
        variants.append([(marker_key, 0, 0), *keyed])
    complete = {}
    for variant in variants:
        phantom = len(variant) - len(keyed)
        phantom_length = sum(len(key) for key, _start, _end in variant[:phantom])
        for first_page in sorted(words_by_page):
            states = {0: [[]]}
            page = first_page
            while states and page in words_by_page:
                following = {}
                for offset, paths in states.items():
                    for count, matched in _note_prefix_matches(
                        variant[offset:], words_by_page[page], body_claimed
                    ):
                        end = offset + count
                        if end <= phantom:
                            continue  # the note marker alone is not text evidence
                        for path in paths:
                            candidate = [*path, (offset, end, matched)]
                            if end == len(variant):
                                if page > first_page:
                                    # A separate renderer label adds no AST
                                    # characters. Plain and labelled matches
                                    # over the SAME glyphs are one path; a
                                    # different occurrence is still ambiguous.
                                    cursor = 0
                                    signature = []
                                    for _lo, _hi, chunk in candidate:
                                        for word in chunk:
                                            next_cursor = cursor + len(word.key)
                                            if next_cursor > phantom_length:
                                                signature.append((
                                                    id(word),
                                                    max(cursor, phantom_length) - phantom_length,
                                                    next_cursor - phantom_length,
                                                ))
                                            cursor = next_cursor
                                    complete.setdefault(tuple(signature), (variant, candidate))
                                    if len(complete) > 1:
                                        return None
                            else:
                                bucket = following.setdefault(end, [])
                                if len(bucket) < 2:
                                    bucket.append(candidate)
                states = following
                page += 1
    if not complete:
        return None
    variant, path = next(iter(complete.values()))
    for start, end, matched in path:
        _consume(paragraph, variant[start:end], matched)
    ordered, matched, total = collect_fragments(attempt)
    ordered.sort(key=lambda item: item[0])
    count, samples = _key_assisted(attempt)
    return (
        ([f for _, group in ordered for f in group], [], matched, total, count, samples, 0),
        frozenset(id(w) for w in paragraph.spelled_words),
    )


def _set_field_phantoms(attempt: _Walker, field_result: Any, page_index: int) -> None:
    """A10/A11: a glyph-bearing paragraph's fields stand in its seams as
    their computed result on THIS page (`_aligner._splice_phantoms`), so
    "34/34" and "Pag. 1 di 2" match exactly and corroborate the paragraph
    before them. A field with no computable result leaves the paragraph
    as it is today."""
    for paragraph in attempt.paragraphs:
        if not paragraph.field_gaps or not _words_with_offsets(paragraph.text):
            continue
        results = [field_result(ft, page_index) for _o, ft, _id in paragraph.field_gaps]
        if all(results):
            paragraph.phantoms = [
                (offset, normalize_for_matching(result)[0])
                for (offset, _ft, _id), result in zip(paragraph.field_gaps, results, strict=True)
            ]


def _tab_segments(
    paragraph: _Paragraph,
    field_result: Any,
    page_index: int,
    unresolved: frozenset[str] = frozenset(),
) -> list[list[tuple[str, int, int]]] | None:
    """A14: the paragraph's keyed words (with each field's computed result
    for this page spliced in, when it has one and its type is not in
    ``unresolved``) cut into segments at every tab and at every glyph gap
    left without a result. ``None`` when such a gap falls inside a word (no
    clean cut, as in `_segments`)."""
    phantoms: list[tuple[int, str]] = []
    if field_result is not None:
        for offset, field_type, _node_id in paragraph.field_gaps:
            if field_type in unresolved:
                continue
            result = field_result(field_type, page_index)
            if result:
                phantoms.append((offset, normalize_for_matching(result)[0]))
    paragraph.phantoms = phantoms
    keyed = _keyed_words(paragraph)
    paragraph.phantoms = []
    covered = {offset for offset, _key in phantoms}
    gaps = [gap for gap in paragraph.glyph_gaps if gap not in covered]
    if any(start < gap < end for _key, start, end in keyed for gap in gaps):
        return None
    segments: list[list[tuple[str, int, int]]] = [[]]
    previous_end = 0
    for key, start, end in keyed:
        if segments[-1] and (
            "\t" in paragraph.text[previous_end:start]
            or any(previous_end <= gap <= start for gap in gaps)
        ):
            segments.append([])
        segments[-1].append((key, start, end))
        previous_end = end
    return [segment for segment in segments if segment]


def _claim_segments(
    paragraphs: list[_Paragraph],
    page_words: list[_StreamWord],
    field_result: Any,
    page_index: int,
    preclaimed: frozenset[int] = frozenset(),
) -> bool:
    """`_claim_segments_once`, then -- when that fails on a paragraph with a
    NUMPAGES field -- once more with NUMPAGES left as a gap: LibreOffice
    counts the blank pages it inserts to keep a restarted page number's
    parity and never exports, so its NUMPAGES can exceed the PDF's page
    count (m2a-v7 a1ee26cb225c prints "of 13" on each of its 12 pages).
    ``preclaimed``: ids of page words another pass already owns."""
    fresh = [p for p in paragraphs if not p.matched]
    if _claim_segments_once(
        paragraphs, page_words, field_result, page_index, preclaimed=preclaimed
    ):
        return True
    if field_result is None or not any(
        field_type == "num_pages" for p in fresh for _o, field_type, _id in p.field_gaps
    ):
        return False
    for paragraph in fresh:
        paragraph.matched = False
        paragraph.matched_words.clear()
        paragraph.spelled_words.clear()
    return _claim_segments_once(
        paragraphs, page_words, field_result, page_index,
        unresolved=frozenset({"num_pages"}), preclaimed=preclaimed,
    )  # fmt: skip


def _claim_segments_once(
    paragraphs: list[_Paragraph],
    page_words: list[_StreamWord],
    field_result: Any,
    page_index: int,
    unresolved: frozenset[str] = frozenset(),
    preclaimed: frozenset[int] = frozenset(),
) -> bool:
    """A14 (policy `story_segment_claims`): claim every glyph-bearing
    paragraph the ordered pass left unmatched, segment by segment
    (`_tab_segments`), in any order. A segment is claimed only when its
    keys are spelled by exactly ONE run of the page's words that no
    paragraph of this story has taken; the rounds repeat while something
    is claimed, so a phrase repeated in two cells ("Fannie Mae" |
    "(c) 2021 Fannie Mae") resolves once the longer one is out. When no
    round claims anything, n still-pending segments with the SAME keys
    that n free, disjoint runs spell are paired in order (document order to
    reading order): the text is the same whichever way they pair (two
    identical form cells "000000000 del /xx/xx/xxxx" in m2a-v7
    1c099565cde9's header table). True only when every segment of every
    such paragraph is claimed -- the story stays all-or-nothing on the
    page.

    pdftotext -layout reorders a line's tab-separated pieces (a right-tab
    "page 23" before the left-aligned "As at 27 Mar 2024") and interleaves
    a footer table's cells; the ordered match cannot survive either."""
    taken = {id(word) for p in paragraphs if p.matched for word in p.spelled_words}
    taken |= preclaimed
    pending: list[tuple[_Paragraph, list[tuple[str, int, int]]]] = []
    for paragraph in paragraphs:
        if paragraph.matched or not _words_with_offsets(paragraph.text):
            continue
        segments = _tab_segments(paragraph, field_result, page_index, unresolved)
        if segments is None:
            return False
        pending += [(paragraph, s) for s in segments if any(key for key, _a, _b in s)]
    if not pending:
        return True
    index = _StreamIndex(page_words)

    def target_of(item: tuple[_Paragraph, list[tuple[str, int, int]]]) -> str:
        return "".join(key for key, _start, _end in item[1])

    def free_runs(target: str, limit: int) -> list[tuple[int, int]]:
        found: list[tuple[int, int]] = []
        for start in index.candidate_starts(target, 0):
            span = _span_at(target, page_words, start)
            if span is None or any(id(w) in taken for w in page_words[span[0] : span[1]]):
                continue
            found.append(span)
            if len(found) > limit:
                break
        return found

    def claim(item: tuple[_Paragraph, list[tuple[str, int, int]]], span: tuple[int, int]) -> None:
        words = page_words[span[0] : span[1]]
        _consume(item[0], item[1], words)
        taken.update(id(word) for word in words)
        pending.remove(item)

    progress = True
    while pending and progress:
        progress = False
        for item in list(pending):
            found = free_runs(target_of(item), 1)
            if len(found) == 1:
                claim(item, found[0])
                progress = True
        if progress or not pending:
            continue
        by_target: dict[str, list[tuple[_Paragraph, list[tuple[str, int, int]]]]] = {}
        for item in pending:
            by_target.setdefault(target_of(item), []).append(item)
        for target, items in by_target.items():
            if len(items) < 2:
                continue
            found = free_runs(target, len(items))
            disjoint = all(a[1] <= b[0] for a, b in zip(found, found[1:], strict=False))
            if len(found) == len(items) and disjoint:
                for item, span in zip(items, found, strict=True):
                    claim(item, span)
                progress = True
    return not pending


# -- repeated table header rows (policy `repeat_header_rows`, A15) ------------------


def _repeating_tables(ast: dict[str, Any]) -> list[tuple[dict[str, Any], str, int]]:
    """`(table, path, n)` for every body table whose first ``n`` rows repeat
    as its header (`layout.repeat_header`) and that has rows after them."""
    out: list[tuple[dict[str, Any], str, int]] = []

    def visit(blocks: list[dict[str, Any]], path: str) -> None:
        for index, block in enumerate(blocks):
            block_path = f"{path}/{index}"
            if block.get("type") == "list":
                for i, item in enumerate(block.get("items", [])):
                    visit(item.get("children", []), f"{block_path}/items/{i}/children")
            elif block.get("type") == "table":
                rows = block.get("rows", [])
                n = 0
                while n < len(rows) and (rows[n].get("layout") or {}).get("repeat_header"):
                    n += 1
                if 0 < n < len(rows):
                    out.append((block, block_path, n))
                for i, row in enumerate(rows):
                    for j, cell in enumerate(row.get("cells", [])):
                        visit(cell.get("children", []), f"{block_path}/rows/{i}/cells/{j}/children")

    for section_index, section in enumerate(ast.get("sections", [])):
        visit(section.get("children", []), f"/sections/{section_index}/children")
    return out


def repeated_header_node_ids(ast: dict[str, Any]) -> frozenset[str]:
    """Every node id inside a repeated header row (A15): observed on each
    page the table continues on, so page-scoped (`chain_scope: "page"`)."""
    from unflatten.dataset.observations.images import _subtree_ids

    ids: set[str] = set()
    for table, _path, n in _repeating_tables(ast):
        for row in table["rows"][:n]:
            ids |= _subtree_ids(row)
    return frozenset(ids)


def align_repeated_header_rows(
    ast: dict[str, Any], pdf_words: list[_StreamWord], body_walker: _Walker
) -> tuple[list[Fragment], set[int]]:
    """A15: Word and LibreOffice paint a table's `repeat_header` rows again
    on every page the table continues on; the body pass spells them once.
    On each page where a later row of the table is observed and no header
    row is, the header rows are claimed segment by segment
    (`_claim_segments`) among the page's words the body pass left free --
    all glyph-bearing header paragraphs or nothing. The rows, their cells,
    paragraphs and runs get one closed chain per page (DEC-033: the page is
    a standalone document and shows the header). Iteration 2 sample: 31
    continuation pages lacked their repeated header, every one flagged.
    Returns the fragments and the ids of the words they spell."""
    words_by_page: dict[int, list[_StreamWord]] = {}
    for word in pdf_words:
        words_by_page.setdefault(word.page_index, []).append(word)
    body_taken = frozenset(id(w) for p in body_walker.paragraphs for w in p.spelled_words)
    pages_of_row: dict[str, set[int]] = {}
    for paragraph in body_walker.paragraphs:
        if paragraph.row_id is not None:
            pages_of_row.setdefault(paragraph.row_id, set()).update(
                w.page_index for w in paragraph.matched_words
            )
    from unflatten.dataset.observations.images import _subtree_ids

    fragments: list[Fragment] = []
    claimed: set[int] = set()
    for table, path, n in _repeating_tables(ast):
        header, rest = table["rows"][:n], table["rows"][n:]
        header_pages = set().union(*(pages_of_row.get(r["id"], set()) for r in header))
        body_pages = set().union(*(pages_of_row.get(r["id"], set()) for r in rest))
        keep = set().union(*(_subtree_ids(r) for r in header))
        for page in sorted(body_pages - header_pages):
            attempt = _Walker()
            try:
                attempt.walk_blocks([{**table, "rows": header}], path)
            except _aligner.TableTopologyError:
                break  # a header cell spans into the body rows: no clean cut
            if not any(_words_with_offsets(p.text) for p in attempt.paragraphs):
                break
            if not _claim_segments(
                attempt.paragraphs, words_by_page.get(page, []), None, page, preclaimed=body_taken
            ):
                continue
            ordered, _matched, _total = collect_fragments(attempt)
            ordered.sort(key=lambda item: item[0])
            fragments += [f for _, group in ordered for f in group if f.node_id in keep]
            claimed |= {id(w) for p in attempt.paragraphs for w in p.spelled_words}
    return fragments, claimed


def _key_assisted(attempt: _Walker) -> tuple[int, list[tuple[str, str]]]:
    """The key-assisted word matches of one successful placement (0g).

    Only the attempt that actually placed the story counts: the failed
    attempts on earlier pages matched words that page does not render,
    and their key flags would be evidence for a placement that never
    happened.
    """
    count = 0
    samples: list[tuple[str, str]] = []
    for paragraph in attempt.paragraphs:
        for word in paragraph.matched_words:
            if not word.key_assisted:
                continue
            count += 1
            if word.raw_pdf is not None:
                samples.append((paragraph.text[word.char_start : word.char_end], word.raw_pdf))
    return count, samples


def _match_across_gaps(
    paragraph: _Paragraph, page_words: list[_StreamWord], *, beyond_window: bool = False
) -> bool:
    """Second chance for a paragraph the page renders glyphs the AST has
    no text for: a `PAGE` field or a note marker.

    A footer reading "Page 3 of 12" is `"Page "` + a `field` + `" of "` +
    a `field` in the AST — DEC-003 §1.0b keeps the RESULT out of the node
    on purpose, because it differs on every page. An exact match over the
    whole paragraph therefore cannot succeed, and `PAGE` was measured in
    34.5 % of the corpus, overwhelmingly in headers and footers. So the
    segments BETWEEN the gaps are matched in order instead, each one
    exactly, with the rendered field glyphs falling in the space between
    them. A word that straddles a gap has no clean segmentation and is
    refused rather than approximated.

    With ``beyond_window`` (DEC-033 mode) the search is not bounded by
    `_RESYNC_WINDOW`, because a footer's glyphs follow the page's body in
    the word stream: beyond the window a segment is accepted only when
    its occurrence is unambiguous and the next segment follows it
    (`_find_char_span`'s resync rule). ``legacy_first_match`` keeps the
    window-only search, because `reading_order.pick_reading_order`
    decides on it and a changed placement there flips whole pages.

    Body paragraphs keep the strict whole-paragraph rule: this is the
    story pass, where an unplaced node costs pixel supervision for a
    whole header.
    """
    segments = _segments(paragraph)
    if segments is None:
        return False
    checkpoint = len(paragraph.matched_words)
    spelled_checkpoint = len(paragraph.spelled_words)
    position = 0
    index = _StreamIndex(page_words) if beyond_window else None
    keys = [[w[0] for w in segment] for segment in segments]
    for number, segment in enumerate(segments):
        following = keys[number + 1] if number + 1 < len(keys) else None
        span = _find_char_span(keys[number], page_words, position, index, following)
        if span is None:
            del paragraph.matched_words[checkpoint:]
            del paragraph.spelled_words[spelled_checkpoint:]
            paragraph.matched = False
            return False
        found, end = span
        _consume(paragraph, segment, page_words[found:end])
        position = end
    return True


def _segments(
    paragraph: _Paragraph,
) -> list[list[tuple[str, int, int]]] | None:
    """The paragraph's words — as pairing keys — grouped between its
    glyph gaps.

    ``None`` when a gap falls INSIDE a word (the rendered glyphs and the
    AST word cannot be separated) or when there is no gap at all — in
    the latter case the strict match already ran and failed, and this
    would just repeat it.
    """
    gaps = paragraph.glyph_gaps
    words = _keyed_words(paragraph)
    if not gaps or not words:
        return None
    if any(start < gap < end for _key, start, end in words for gap in gaps):
        return None
    segments: list[list[tuple[str, int, int]]] = [[]]
    for word in words:
        _key, start, _end = word
        if segments[-1] and any(segments[-1][-1][2] <= gap <= start for gap in gaps):
            segments.append([])
        segments[-1].append(word)
    return [segment for segment in segments if segment]


def _walk(story: _Story) -> _Walker:
    walker = _Walker()
    walker.walk_blocks(story.node.get("children", []), story.path)
    return walker
