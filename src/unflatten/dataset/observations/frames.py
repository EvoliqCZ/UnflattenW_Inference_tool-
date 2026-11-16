"""Per-page alignment of `text_frame` content (DEC-033 §7).

A text frame is an anchored block-content container (RFC001-ERRATA-004
E4) that can float anywhere on the page it is anchored to — unlike a
header/footer, it renders once, but its position relative to the body
stream is not fixed the way a body paragraph's is, so the single
sequential body pass in :mod:`unflatten.dataset.observations.aligner`
(which never descends into a `text_frame`, aligner.py:585) cannot place
it. Before this module a frame's runs reached the Observation Sidecar
with no observations at all — `unresolved` page evidence, withdrawn as
`not_supervised` (DEC-033 §7 diagnosis).

**The rule (DEC-033 §7):** a frame is placed on the first page whose
words spell every one of its glyph-bearing paragraphs — a single
placement, `chain_scope: "document"` (the default), because a frame
renders once, unlike a repeated header. Candidate pages are tried in an
order biased toward where the frame most likely sits: the page of the
last matched word of the nearest preceding body paragraph IN THE SAME
SECTION first, then every page in document order. That anchor is only
computed for frames sitting directly in a section's body; a frame
inside a header/footer/note story has no comparable position and
always falls back to the full page sweep — a deliberate scope
narrowing (most frames are body frames) rather than an attempt to
derive a proximity signal a story has no AST position for.

A frame that matches no page yields `OBSERVATION_FRAME_UNPLACED` and no
observations, the same shape as `OBSERVATION_STORY_UNPLACED`.

**The frame node's own box** (`frame_observations`, 2026-09-26): the
placement above grants boxes to the frame's paragraphs and runs, but
the `text_frame` node itself reached the sidecar with none — so the
model's region target for a frame was always empty (`regions []`)
and it never learned where a frame sits. The frame's observation is
derived, never matched: on every page where the frame's content has
observations (text fragments from this pass, image fragments from
:mod:`images`), the frame's box is the union of those content boxes —
the same convention by which a paragraph's box is the union of its
words' boxes, i.e. the ink extent, not Word's frame rectangle with its
padding. It carries no char range (the frame node owns no text) and is
tagged with the RFC003 §24.4 mapping method `geometry_inferred`, so a
reader can tell a derived frame box from a text match or an image
object match. Under DEC-033 §7 a frame lands on exactly one page, so
the chain is normally one fragment; the per-page union is defined
anyway (an image inside the frame is placed by its own rule), and a
multi-page chain follows the text convention (`fragment_index` in page
order with continuation flags), or one closed chain per page for a
page-scoped frame (`chain_scope: "page"`, DEC-033 §3).

Because a frame's placement attempt runs its own independent
`_match_paragraphs` call against a page's words (a fresh search from
position 0, like the story pass), it can never "steal" words that the
body pass consumed, and the body pass can never see a frame's runs
either — the two passes read the same page word list but do not share
matching state. Frame text that happens to duplicate a body paragraph
is therefore placed correctly by both passes independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import unflatten.dataset.observations.aligner as _aligner
from unflatten.ast.issues import Issue
from unflatten.dataset.observations.aligner import (
    Fragment,
    _has_glyphs,
    _StreamWord,
    _Walker,
    _words_with_offsets,
    collect_fragments,
)
from unflatten.dataset.observations.images import _subtree_ids
from unflatten.dataset.observations.stories import (
    _claim_segments,
    _match_across_gaps,
    _set_field_phantoms,
)
from unflatten.dataset.targets.events import NOTE_KEYS

_SECTION_STORY_KEYS: tuple[str, ...] = ("headers", "footers")


@dataclass
class FrameAlignment:
    """Per-page frame alignment result, in `AlignmentResult` shape plus
    the DEC-033 §7 coverage counters."""

    fragments: list[Fragment]
    issues: list[Issue]
    matched_text_nodes: int
    total_text_nodes: int
    frames_total: int = 0
    frames_placed: int = 0
    frame_runs_total: int = 0
    frame_runs_observed: int = 0
    #: matched characters across observed frame runs — the amount of
    #: text a placed frame actually grants, not just a placed/unplaced
    #: count
    frame_text_scalars: int = 0


@dataclass(frozen=True)
class FrameFragment:
    """One observation of a `text_frame` node itself on one page: the
    union box of its placed content on that page (module docstring).
    Same surface as `images.ImageFragment` — no char range."""

    node_id: str
    page_index: int
    bbox_ppm: tuple[int, int, int, int]
    fragment_index: int = 0
    continuation_prev: bool = False
    continuation_next: bool = False


@dataclass(frozen=True)
class _Frame:
    node: dict[str, Any]
    path: str


def frames_of(ast: dict[str, Any]) -> list[_Frame]:
    """Every `text_frame` node in `ast`, wherever a block node may sit:
    a section's own body, a header/footer/note story, or nested inside
    a list item or table cell of any of those."""
    frames: list[_Frame] = []
    for section_index, section in enumerate(ast.get("sections", [])):
        _collect_frames(
            section.get("children", []), f"/sections/{section_index}/children", frames
        )
        for key in _SECTION_STORY_KEYS:
            for story_index, story in enumerate(section.get(key) or []):
                _collect_frames(
                    story.get("children", []),
                    f"/sections/{section_index}/{key}/{story_index}/children",
                    frames,
                )
    for key in NOTE_KEYS:
        for note_index, note in enumerate(ast.get(key) or []):
            _collect_frames(
                note.get("children", []), f"/{key}/{note_index}/children", frames
            )
    return frames


def _collect_frames(blocks: list[dict[str, Any]], path: str, frames: list[_Frame]) -> None:
    """Recurse into `blocks` (a `children`/`items`/`cells` array) looking
    for `text_frame` nodes, descending into `list` and `table`
    structure the same way `_Walker._block` does. Nested text frames are
    schema-representable but semantically rejected, so this does not
    recurse into a found frame's own children."""
    for index, block in enumerate(blocks):
        block_path = f"{path}/{index}"
        block_type = block.get("type")
        if block_type == "text_frame":
            frames.append(_Frame(block, block_path))
        elif block_type == "list":
            for i, item in enumerate(block.get("items", [])):
                _collect_frames(
                    item.get("children", []), f"{block_path}/items/{i}/children", frames
                )
        elif block_type == "table":
            for i, row in enumerate(block.get("rows", [])):
                for j, cell in enumerate(row.get("cells", [])):
                    _collect_frames(
                        cell.get("children", []),
                        f"{block_path}/rows/{i}/cells/{j}/children",
                        frames,
                    )
        # paragraph / page_break / column_break: no frames inside


def _anchor_pages(ast: dict[str, Any], body_walker: _Walker) -> dict[str, int]:
    """frame node id -> the page of the last matched word of the
    nearest preceding body paragraph in the frame's own section (the
    proximity heuristic, module docstring). A frame with no such
    paragraph, or whose anchor paragraph never matched, is absent from
    the returned mapping and falls back to the full page sweep."""
    last_page_of: dict[str, int] = {}
    for paragraph in body_walker.paragraphs:
        if paragraph.matched_words:
            last_page_of[paragraph.node_id] = max(
                w.page_index for w in paragraph.matched_words
            )

    anchor_paragraph: dict[str, str] = {}

    def walk(blocks: list[dict[str, Any]], last: str | None) -> str | None:
        for block in blocks:
            block_type = block.get("type")
            if block_type == "paragraph":
                last = block["id"]
            elif block_type == "list":
                for item in block.get("items", []):
                    last = walk(item.get("children", []), last)
            elif block_type == "table":
                for row in block.get("rows", []):
                    for cell in row.get("cells", []):
                        last = walk(cell.get("children", []), last)
            elif block_type == "text_frame" and last is not None:
                anchor_paragraph[block["id"]] = last
        return last

    for section in ast.get("sections", []):
        walk(section.get("children", []), None)

    return {
        frame_id: last_page_of[paragraph_id]
        for frame_id, paragraph_id in anchor_paragraph.items()
        if paragraph_id in last_page_of
    }


def _story_path(frame_path: str) -> str | None:
    """`/sections/N/headers|footers/i` when the frame sits in that story."""
    parts = frame_path.split("/")
    if len(parts) > 4 and parts[1] == "sections" and parts[3] in _SECTION_STORY_KEYS:
        return "/".join(parts[:5])
    return None


def align_frames(
    ast: dict[str, Any],
    pdf_words: list[_StreamWord],
    body_walker: _Walker,
    *,
    story_pages: dict[str, list[int]] | None = None,
    field_result: Any = None,
    segment_claims: bool = False,
) -> FrameAlignment:
    """Observe every `text_frame` on the first page whose words spell it
    (DEC-033 §7). `pdf_words` is the whole document's keyed word stream,
    shared with the body and story passes; `body_walker` is the body
    pass's own `_Walker` (already matched), used only for the proximity
    ordering (`_anchor_pages`).

    ``story_pages`` (policy `v16-align.v1`, A13): a frame inside a
    header/footer whose story path is a key is placed on EVERY one of that
    story's pages its words spell (DEC-033 §3, like the story itself),
    with the story's field results (``field_result``) and, with
    ``segment_claims``, the A14 segment claims."""
    words_by_page: dict[int, list[_StreamWord]] = {}
    for word in pdf_words:
        words_by_page.setdefault(word.page_index, []).append(word)
    all_pages = sorted(words_by_page)
    anchor_pages = _anchor_pages(ast, body_walker)

    fragments: list[Fragment] = []
    issues: list[Issue] = []
    matched_nodes = 0
    total_nodes = 0
    frames_total = 0
    frames_placed = 0
    frame_runs_total = 0
    frame_runs_observed = 0
    frame_text_scalars = 0

    for frame in frames_of(ast):
        frames_total += 1
        anchor = anchor_pages.get(frame.node["id"])
        candidate_pages = (
            [anchor, *(p for p in all_pages if p != anchor)] if anchor is not None else all_pages
        )
        story = _story_path(frame.path) if story_pages is not None else None
        keep_all = story is not None and story in story_pages
        if keep_all:
            candidate_pages = story_pages[story]
        (
            placed,
            frame_issues,
            matched,
            total,
            runs_total,
            runs_observed,
            scalars,
        ) = _place_frame(
            frame, words_by_page, candidate_pages,
            keep_all=keep_all,
            field_result=field_result if keep_all else None,
            segment_claims=segment_claims and keep_all,
        )  # fmt: skip
        fragments.extend(placed)
        issues.extend(frame_issues)
        matched_nodes += matched
        total_nodes += total
        frame_runs_total += runs_total
        frame_runs_observed += runs_observed
        frame_text_scalars += scalars
        if placed:
            frames_placed += 1

    return FrameAlignment(
        fragments=fragments,
        issues=issues,
        matched_text_nodes=matched_nodes,
        total_text_nodes=total_nodes,
        frames_total=frames_total,
        frames_placed=frames_placed,
        frame_runs_total=frame_runs_total,
        frame_runs_observed=frame_runs_observed,
        frame_text_scalars=frame_text_scalars,
    )


def _walk(frame: _Frame) -> _Walker:
    walker = _Walker()
    walker.walk_blocks(frame.node.get("children", []), frame.path)
    return walker


def _run_glyph_count(walker: _Walker) -> int:
    """Glyph-bearing runs in `walker`'s paragraphs, independent of any
    match attempt — the denominator for `frame_runs_observed`."""
    count = 0
    for paragraph in walker.paragraphs:
        for _run_id, _preorder, start, end in paragraph.run_spans:
            if _has_glyphs(paragraph.text[start:end]):
                count += 1
    return count


def _observed_run_stats(walker: _Walker) -> tuple[int, int]:
    """(runs with at least one matched word, matched characters) over an
    ALREADY-MATCHED `walker`'s glyph-bearing runs."""
    runs_observed = 0
    scalars = 0
    for paragraph in walker.paragraphs:
        for _run_id, _preorder, start, end in paragraph.run_spans:
            if not _has_glyphs(paragraph.text[start:end]):
                continue
            run_words = [
                w for w in paragraph.matched_words if w.char_start < end and w.char_end > start
            ]
            if not run_words:
                continue
            runs_observed += 1
            scalars += sum(
                min(w.char_end, end) - max(w.char_start, start) for w in run_words
            )
    return runs_observed, scalars


def _place_frame(
    frame: _Frame,
    words_by_page: dict[int, list[_StreamWord]],
    candidate_pages: list[int],
    *,
    keep_all: bool = False,
    field_result: Any = None,
    segment_claims: bool = False,
) -> tuple[list[Fragment], list[Issue], int, int, int, int, int]:
    """``keep_all`` (A13, a frame inside a header/footer): every candidate
    page whose words spell the frame is kept, not only the first; the
    counters then sum over the placed pages, like a story's."""
    base_walker = _walk(frame)
    with_words = [p for p in base_walker.paragraphs if _words_with_offsets(p.text)]
    runs_total = _run_glyph_count(base_walker)
    if keep_all:
        # Observed runs are counted per placement. Count the denominator
        # per applicable story page too, including pages with no match.
        runs_total *= max(1, len(candidate_pages))
    if not with_words:
        # image-only / empty frame: nothing to search for
        return [], [], 0, 0, runs_total, 0, 0

    kept: list[Fragment] = []
    kept_matched = kept_total = kept_runs = kept_scalars = 0
    for page_index in candidate_pages:
        attempt = _walk(frame)
        page_words = words_by_page.get(page_index, [])
        if field_result is not None:
            _set_field_phantoms(attempt, field_result, page_index)
        _aligner._match_paragraphs(attempt.paragraphs, page_words, [])
        for paragraph in attempt.paragraphs:
            paragraph.phantoms = []
        if not all(
            paragraph.matched or _match_across_gaps(paragraph, page_words)
            for paragraph in attempt.paragraphs
            if _words_with_offsets(paragraph.text)
        ) and not (
            segment_claims
            and _claim_segments(attempt.paragraphs, page_words, field_result, page_index)
        ):
            continue
        ordered, matched, total = collect_fragments(attempt)
        ordered.sort(key=lambda item: item[0])
        fragments = [fragment for _, group in ordered for fragment in group]
        runs_observed, scalars = _observed_run_stats(attempt)
        if not keep_all:
            return fragments, [], matched, total, runs_total, runs_observed, scalars
        kept.extend(fragments)
        kept_matched += matched
        kept_total += total
        kept_runs += runs_observed
        kept_scalars += scalars
    if kept:
        return kept, [], kept_matched, kept_total, runs_total, kept_runs, kept_scalars

    _unplaced, _matched, total = collect_fragments(base_walker)
    return (
        [],
        [
            Issue(
                code="OBSERVATION_FRAME_UNPLACED",
                severity="error",
                node_id=frame.node["id"],
                path=frame.path,
                message=(
                    f"{frame.node['type']} text was not found on any rendered page; "
                    "no page is claimed for it (DEC-033 §7)."
                ),
                details={"pages_searched": candidate_pages},
            )
        ],
        0,
        total,
        runs_total,
        0,
        0,
    )


def frame_observations(
    ast: dict[str, Any],
    fragments: list[Any],
    page_scoped_ids: frozenset[str] = frozenset(),
) -> list[FrameFragment]:
    """The frame node's own box on every page where its content was
    observed: the union of the content boxes there (module docstring).

    `fragments` is every placed observation of the document — the
    aligner's text `Fragment`s and the image pass's `ImageFragment`s,
    read through their common `node_id`/`page_index`/`bbox_ppm`
    surface. Only fragments of the frame's DESCENDANTS count; a
    fragment carrying the frame's own id is never read back, so the
    frame box is always derived, never copied. A frame with no observed
    content (unplaced, or image-only with an unplaced image) yields
    nothing — an empty box would be an invented label.

    Frames in `page_scoped_ids` (inside a header/footer, DEC-033 §3)
    get one closed chain per page; every other frame gets one document
    chain in page order.
    """
    boxes_by_node: dict[str, list[tuple[int, tuple[int, int, int, int]]]] = {}
    for fragment in fragments:
        boxes_by_node.setdefault(fragment.node_id, []).append(
            (fragment.page_index, tuple(fragment.bbox_ppm))
        )

    observations: list[FrameFragment] = []
    for frame in frames_of(ast):
        frame_id = frame.node["id"]
        by_page: dict[int, list[tuple[int, int, int, int]]] = {}
        for node_id in _subtree_ids(frame.node) - {frame_id}:
            for page_index, bbox in boxes_by_node.get(node_id, ()):
                by_page.setdefault(page_index, []).append(bbox)
        pages = sorted(by_page)
        page_scoped = frame_id in page_scoped_ids
        for k, page_index in enumerate(pages):
            boxes = by_page[page_index]
            union = (
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
            )
            observations.append(
                FrameFragment(
                    node_id=frame_id,
                    page_index=page_index,
                    bbox_ppm=union,
                    fragment_index=0 if page_scoped else k,
                    continuation_prev=False if page_scoped else k > 0,
                    continuation_next=False if page_scoped else k < len(pages) - 1,
                )
            )
    return observations
